import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import rasterio
from rasterio.transform import Affine
from rasterio.features import shapes
from rasterio.windows import Window
from rasterio.crs import CRS
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon, box
from ultralytics import YOLO
import torch

class YOLOGeoTIFFInference:
    """
    Performs YOLO inference on large GeoTIFF images using patch-based processing.
    Uses original NIR bands (no CHM).
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        overlap_ratio: float = 0.2,
        confidence_threshold: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.device = device

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"Model loaded successfully on device: {device}")

    def _get_image_patches_generator(self, nir_src: rasterio.io.DatasetReader, patch_size: int):
        height, width = nir_src.height, nir_src.width
        stride = int(patch_size * (1 - self.overlap_ratio))

        for y_start in range(0, height, stride):
            for x_start in range(0, width, stride):
                y_end = min(y_start + patch_size, height)
                x_end = min(x_start + patch_size, width)

                nir_window = Window(col_off=x_start, row_off=y_start, width=x_end-x_start, height=y_end-y_start)
                nir_patch_data = nir_src.read([1, 2, 3], window=nir_window) # Lê apenas R, G, B
                
                # Transpor para (H, W, C)
                composed_patch = np.transpose(nir_patch_data, (1, 2, 0))

                # Preencher com zeros se o patch for menor que patch_size
                if composed_patch.shape[0] < patch_size or composed_patch.shape[1] < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=nir_patch_data.dtype)
                    padded_patch[:composed_patch.shape[0], :composed_patch.shape[1]] = composed_patch
                    composed_patch = padded_patch

                window_transform = rasterio.windows.transform(nir_window, nir_src.transform)

                yield {
                    'data': composed_patch,
                    'y_start': y_start, 'y_end': y_end,
                    'x_start': x_start, 'x_end': x_end,
                    'window_transform': window_transform
                }

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def vectorize_mask(self, mask: np.ndarray, transform: Affine, crs: CRS, min_area_pixels: int) -> gpd.GeoDataFrame:
        polygons = []
        for geom, val in shapes(mask, transform=transform):
            if val == 1:
                poly = shape(geom)
                if poly.area >= min_area_pixels:
                    polygons.append(poly)
        
        if not polygons:
            return gpd.GeoDataFrame(geometry=[], crs=crs)

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        return gdf

    def process(self, geotiff_path: str, output_path: str, min_area_pixels: int = 10, output_format: str = "geopackage") -> None:
        print("\n" + "="*60 + "\nYOLO NIR Inference Pipeline\n" + "="*60 + "\n")
        
        with rasterio.open(geotiff_path) as nir_src:
            metadata = {
                'crs': nir_src.crs,
                'transform': nir_src.transform,
                'width': nir_src.width,
                'height': nir_src.height
            }
            print(f"GeoTIFF NIR loaded: ({nir_src.height}, {nir_src.width}), CRS: {metadata['crs']}")

            all_gdfs = []
            class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            # Calcular total de patches
            stride = int(self.imgsz * (1 - self.overlap_ratio))
            num_y_patches = (nir_src.height + stride - 1) // stride
            num_x_patches = (nir_src.width + stride - 1) // stride
            total_patches = num_y_patches * num_x_patches

            print(f"Running inference on {total_patches} patches...")

            patch_idx = 0
            for patch_info in self._get_image_patches_generator(nir_src, self.imgsz):
                patch_idx += 1
                if (patch_idx % max(1, total_patches // 10)) == 0 or patch_idx == 1 or patch_idx == total_patches:
                    print(f"  Processing patch {patch_idx}/{total_patches}")

                if np.all(patch_info['data'] == 0):
                    continue

                results = self.model(
                    patch_info['data'],
                    imgsz=self.imgsz,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                if results[0].masks is not None:
                    masks_data = results[0].masks.data
                    if masks_data is not None and masks_data.numel() > 0:
                        masks = (masks_data.cpu().numpy() > 0.5).astype(np.uint8)
                        
                        for i in range(masks.shape[0]):
                            class_id = int(results[0].boxes.cls[i].item())
                            mask = masks[i]

                            cleaned_mask = self._clean_mask(mask)
                            gdf = self.vectorize_mask(cleaned_mask, patch_info["window_transform"], metadata["crs"], min_area_pixels)
                            
                            if not gdf.empty:
                                # Extrair bounding box
                                bbox_coords = results[0].boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = bbox_coords
                                
                                lon1, lat1 = rasterio.transform.xy(patch_info["window_transform"], y1, x1)
                                lon2, lat2 = rasterio.transform.xy(patch_info["window_transform"], y2, x2)
                                bbox_geom = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

                                gdf["class_id"] = class_id
                                gdf["class_name"] = class_names.get(class_id, f"class_{class_id}")
                                gdf["bbox_geometry"] = bbox_geom
                                all_gdfs.append(gdf)

            if not all_gdfs:
                print("Nenhuma detecção encontrada.")
                return

            final_gdf = pd.concat(all_gdfs, ignore_index=True)
            
            # Criar GeoDataFrames separados
            gdf_segmentation = gpd.GeoDataFrame(final_gdf.drop(columns=['bbox_geometry']), geometry='geometry', crs=metadata["crs"])
            gdf_bboxes = gpd.GeoDataFrame(final_gdf.drop(columns=['geometry']), geometry='bbox_geometry', crs=metadata["crs"])
            gdf_bboxes = gdf_bboxes.rename_geometry('geometry')

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_output_name = os.path.splitext(output_path)[0]
            ext = ".gpkg" if output_format == "geopackage" else ".shp"
            driver = "GPKG" if output_format == "geopackage" else "ESRI Shapefile"

            path_seg = f"{base_output_name}_segmentation{ext}"
            path_bbox = f"{base_output_name}_bboxes{ext}"

            gdf_segmentation.to_file(path_seg, driver=driver)
            gdf_bboxes.to_file(path_bbox, driver=driver)

            print(f"Resultados salvos em:\n - {path_seg}\n - {path_bbox}")

def main():
    MODEL_PATH = r"C:\araucaria_yolo\runs\segment\experiment_10_NIR\weights\best.pt"
    GEOTIFF_PATH = r"C:\araucaria_yolo\IMAGEM_IR\SF-23-Y-B-V-2-SE-D_ORTO_IR.tif"
    OUTPUT_PATH = r"C:\araucaria_yolo\predictions\prediction_output_nir.gpkg"

    inference = YOLOGeoTIFFInference(
        model_path=MODEL_PATH,
        imgsz=640,
        overlap_ratio=0.8,
        confidence_threshold=0.5
    )

    inference.process(
        geotiff_path=GEOTIFF_PATH,
        output_path=OUTPUT_PATH,
        min_area_pixels=10,
        output_format="geopackage"
    )

if __name__ == '__main__':
    main()
