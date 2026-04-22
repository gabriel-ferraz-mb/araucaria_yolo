import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.features import shapes
    from rasterio.windows import Window
    from rasterio.warp import reproject, Resampling
    from rasterio.merge import merge
    from rasterio.crs import CRS
except ImportError:
    print("Installing rasterio...")
    os.system("sudo pip3 install rasterio -q")
    import rasterio
    from rasterio.transform import Affine
    from rasterio.features import shapes
    from rasterio.windows import Window
    from rasterio.warp import reproject, Resampling
    from rasterio.merge import merge
    from rasterio.crs import CRS

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import shape, Polygon, MultiPolygon, box
except ImportError:
    print("Installing geopandas and shapely...")
    os.system("sudo pip3 install geopandas pandas shapely -q")
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import shape, Polygon, MultiPolygon, box

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("sudo pip3 install ultralytics -q")
    import ultralytics
    from ultralytics import YOLO

try:
    import torch
except ImportError:
    print("Installing torch...")
    os.system("sudo pip3 install torch -q")
    import torch

import re

def normalize_chm(chm_array, max_height=30.0):
    """
    Normaliza os valores de altura (metros) para o intervalo 0-255.
    Valores <= 0 ficam 0, valores >= max_height ficam 255.
    """
    chm_norm = np.clip(chm_array, 0, max_height)
    chm_norm = (chm_norm / max_height) * 255
    return chm_norm.astype(np.uint8)

def get_merged_chm_mosaic(rgb_base_name_original: str, input_chm_dir: str, default_chm_crs: int) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Localiza os quadrantes CHM (I, II, III, IV) para um dado RGB, faz o merge e
    retorna os dados do mosaico e seus metadados.
    """
    # Extrai o nome base do RGB removendo sufixos conhecidos
    root_name = rgb_base_name_original.replace('_ORTO_RGB', '')
    
    print(f"  DEBUG: Nome base extraído do RGB: {root_name}")

    chm_suffixes = ["I", "II", "III", "IV"]
    src_files_to_merge = []
    
    for suffix in chm_suffixes:
        chm_filename = f"{root_name}-{suffix}_chm.tif"
        chm_path = os.path.join(input_chm_dir, chm_filename)
        if os.path.exists(chm_path):
            try:
                src_chm = rasterio.open(chm_path)
                # Se o CHM não tem CRS, atribui o default_chm_crs para merge
                if not src_chm.crs:
                    print(f"  Aviso: CHM {chm_filename} sem CRS. Atribuindo EPSG:{default_chm_crs}.")
                    # Cria um MemoryFile para atribuir o CRS sem modificar o original
                    profile = src_chm.profile
                    profile.update(crs=CRS.from_epsg(default_chm_crs))
                    memfile = rasterio.MemoryFile()
                    with memfile.open(**profile) as dst:
                        dst.write(src_chm.read())
                    src_files_to_merge.append(memfile.open())
                    src_chm.close() # Fecha o original
                else:
                    src_files_to_merge.append(src_chm)
                print(f"  DEBUG: CHM encontrado e adicionado para merge: {chm_filename}")
            except Exception as e:
                print(f"  Erro ao abrir CHM {chm_filename}: {e}")
        else:
            print(f"  DEBUG: CHM não encontrado: {chm_filename}")

    if not src_files_to_merge:
        print(f"  Aviso: Nenhum CHM válido encontrado para {rgb_base_name_original}. Pulando CHM.")
        return None, None

    try:
        mosaic, out_transform = merge(src_files_to_merge)
        out_meta = src_files_to_merge[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "crs": src_files_to_merge[0].crs # Usa o CRS do primeiro arquivo (ou do atribuído)
        })
        print(f"  DEBUG: Mosaico CHM criado com sucesso. Dimensões: {mosaic.shape}, CRS: {out_meta['crs']}")
        return mosaic[0], out_meta # Retorna apenas a primeira banda do mosaico
    except Exception as e:
        print(f"  Erro ao fazer mosaico dos CHMs para {rgb_base_name_original}: {e}")
        return None, None
    finally:
        for src in src_files_to_merge:
            src.close()


class YOLOGeoTIFFInference:
    """
    Performs YOLO inference on large GeoTIFF images using patch-based processing.
    Combines masks and converts results to vector format (GeoPackage or Shapefile).
    """

    def __init__(
        self,
        model_path: str,
        chm_folder: str,
        default_chm_crs: int = 31983,
        imgsz: int = 640,
        overlap_ratio: float = 0.2,
        confidence_threshold: float = 0.8,
        device: str = "cuda" 
    ):
        self.model_path = model_path
        self.chm_folder = chm_folder
        self.default_chm_crs = default_chm_crs
        self.imgsz = imgsz
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.device = device

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"Model loaded successfully on device: {device}")

    def _get_image_patches_generator(self, rgb_src: rasterio.io.DatasetReader, chm_mosaic_data: np.ndarray, chm_mosaic_meta: Dict, patch_size: int) -> Tuple[np.ndarray, Dict]:
        height, width = rgb_src.height, rgb_src.width
        stride = int(patch_size * (1 - self.overlap_ratio))

        for y_start in range(0, height, stride):
            for x_start in range(0, width, stride):
                # Ajustar as janelas para não exceder os limites da imagem
                y_end = min(y_start + patch_size, height)
                x_end = min(x_start + patch_size, width)

                # Criar a janela de leitura para o RGB
                rgb_window = Window(col_off=x_start, row_off=y_start, width=x_end-x_start, height=y_end-y_start)
                rgb_patch_data = rgb_src.read(window=rgb_window)

                # Compor o patch de 3 bandas (CHM, G, B)
                # Reamostrar CHM para as dimensões exatas do patch RGB se necessário
                chm_patch_data = np.empty((1, rgb_patch_data.shape[1], rgb_patch_data.shape[2]), dtype=chm_mosaic_data.dtype)
                
                # Calcular a transformação para a janela do CHM
                chm_window_transform = rasterio.windows.transform(rgb_window, rgb_src.transform)

                reproject(
                    source=chm_mosaic_data,
                    destination=chm_patch_data,
                    src_transform=chm_mosaic_meta["transform"],
                    src_crs=chm_mosaic_meta["crs"],
                    dst_transform=chm_window_transform,
                    dst_crs=rgb_src.crs,
                    resampling=Resampling.bilinear
                )
                chm_patch_data = chm_patch_data[0] # Remover dimensão da banda

                # Normalizar CHM e compor (CHM, G, B)
                chm_8bit = normalize_chm(chm_patch_data, max_height=35.0)
                composed_patch = np.stack((chm_8bit, rgb_patch_data[1], rgb_patch_data[2]), axis=0)

                # Transpor para (H, W, C) para o YOLO
                composed_patch = np.transpose(composed_patch, (1, 2, 0))

                # Preencher com zeros se o patch for menor que patch_size (bordas)
                if composed_patch.shape[0] < patch_size or composed_patch.shape[1] < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded_patch[:composed_patch.shape[0], :composed_patch.shape[1]] = composed_patch
                    composed_patch = padded_patch

                yield {
                    'data': composed_patch,
                    'y_start': y_start, 'y_end': y_end,
                    'x_start': x_start, 'x_end': x_end,
                    'original_height': y_end - y_start,
                    'original_width': x_end - x_start,
                    'window_transform': chm_window_transform # Usar a transformação da janela para vetorização
                }

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        # Aplica operações morfológicas para limpar a máscara
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fecha pequenos buracos
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove pequenos ruídos
        return mask

    def vectorize_mask(self, mask: np.ndarray, transform: Affine, crs: CRS, min_area_pixels: int) -> gpd.GeoDataFrame:
        polygons = []
        for geom, val in shapes(mask, transform=transform):
            if val == 1:  # Se for parte da máscara
                poly = shape(geom)
                if poly.area >= min_area_pixels:
                    polygons.append(poly)
        
        if not polygons:
            return gpd.GeoDataFrame(geometry=[], crs=crs)

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        return gdf

    def process(self, geotiff_path: str, output_path: str, min_area_pixels: int = 10, output_format: str = "geopackage") -> None:
        print("\n" + "="*60 + "\nYOLO GeoTIFF Inference Pipeline\n" + "="*60 + "\n")
        
        rgb_base_name = Path(geotiff_path).stem # Nome base do RGB sem extensão

        # 1. Carregar CHM e criar mosaico
        print(f"Preparando mosaico CHM para {rgb_base_name}...")
        chm_mosaic_data, chm_mosaic_meta = get_merged_chm_mosaic(rgb_base_name, self.chm_folder, self.default_chm_crs)
        if chm_mosaic_data is None:
            print("Erro: Não foi possível preparar o mosaico CHM. Abortando inferência.")
            return

        # 2. Abrir RGB
        print(f"Loading GeoTIFF RGB from: {geotiff_path}")
        with rasterio.open(geotiff_path) as rgb_src:
            metadata = {
                'crs': rgb_src.crs,
                'transform': rgb_src.transform,
                'width': rgb_src.width,
                'height': rgb_src.height,
                'count': 3, # Será 3 bandas (CHM, G, B)
                'dtype': np.uint8, # Será uint8 após normalização
                'bounds': rgb_src.bounds,
                'res': rgb_src.res
            }
            print(f"GeoTIFF RGB loaded: ({rgb_src.height}, {rgb_src.width}, {rgb_src.count}), CRS: {metadata['crs']}")

            all_gdfs = []
            class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            # Calcular total_patches corretamente para a barra de progresso
            height, width = rgb_src.height, rgb_src.width
            stride = int(self.imgsz * (1 - self.overlap_ratio))
            num_y_patches = (height + stride - 1) // stride # Equivalente a ceil(height / stride)
            num_x_patches = (width + stride - 1) // stride # Equivalente a ceil(width / stride)
            total_patches = num_y_patches * num_x_patches

            print(f"Running inference on {total_patches} patches...")

            patch_idx = 0
            for patch_info in self._get_image_patches_generator(rgb_src, chm_mosaic_data, chm_mosaic_meta, self.imgsz):
                patch_idx += 1
                if (patch_idx % max(1, total_patches // 10)) == 0 or patch_idx == 1 or patch_idx == total_patches:
                    print(f"  Processing patch {patch_idx}/{total_patches}")

                # Otimização: pular patches que são completamente pretos (sem dados)
                if np.all(patch_info['data'] == 0):
                    continue

                results = self.model(
                    patch_info['data'],
                    imgsz=self.imgsz,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                # Correção: verificar se results[0].masks não é None antes de acessar .data
                if results[0].masks is not None:
                    masks_data = results[0].masks.data
                    # No PyTorch, .size é um método, por isso usamos .numel() para verificar se há elementos
                    if masks_data is not None and masks_data.numel() > 0:
                        masks = (masks_data.cpu().numpy() > 0.5).astype(np.uint8)
                        
                        for i in range(masks.shape[0]):
                            class_id = int(results[0].boxes.cls[i].item())
                            mask = masks[i]

                            # Limpar e vetorizar a máscara diretamente
                            cleaned_mask = self._clean_mask(mask)
                            gdf = self.vectorize_mask(cleaned_mask, patch_info["window_transform"], metadata["crs"], min_area_pixels)
                            
                            if not gdf.empty:
                                # Extrair bounding box
                                bbox_coords = results[0].boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = bbox_coords

                                # Converter coordenadas do bbox para o sistema de coordenadas do patch
                                # A window_transform mapeia coordenadas de pixel para coordenadas geográficas
                                # As coordenadas xyxy são relativas ao patch (0,0) até (imgsz, imgsz)
                                # Precisamos mapear (x1, y1) e (x2, y2) para o CRS do patch
                                
                                # Obter as coordenadas geográficas dos cantos do bbox
                                lon1, lat1 = rasterio.transform.xy(patch_info["window_transform"], y1, x1)
                                lon2, lat2 = rasterio.transform.xy(patch_info["window_transform"], y2, x2)

                                # Criar a geometria do bounding box
                                bbox_geom = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

                                gdf["class_id"] = class_id
                                gdf["class_name"] = class_names.get(class_id, f"class_{class_id}")
                                gdf["bbox_geometry"] = bbox_geom # Adiciona a geometria do bbox
                                all_gdfs.append(gdf)

            if not all_gdfs:
                print("Nenhuma detecção encontrada. Nenhum arquivo vetorial será gerado.")
                return

            final_gdf = pd.concat(all_gdfs, ignore_index=True)
            
            # Criar GeoDataFrame para Segmentação
            # A coluna 'geometry' já é a geometria ativa do GeoDataFrame original
            gdf_segmentation = gpd.GeoDataFrame(
                final_gdf.drop(columns=['bbox_geometry']),
                geometry='geometry',
                crs=metadata["crs"]
            )

            # Criar GeoDataFrame para Bounding Boxes
            # Usamos a coluna 'bbox_geometry' como a geometria ativa
            gdf_bboxes = gpd.GeoDataFrame(
                final_gdf.drop(columns=['geometry']),
                geometry='bbox_geometry',
                crs=metadata["crs"]
            )
            # Renomear a coluna de geometria para o padrão 'geometry' para compatibilidade máxima
            gdf_bboxes = gdf_bboxes.rename_geometry('geometry')

            # Salvar resultados
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_output_name = os.path.splitext(output_path)[0]
            ext = ".gpkg" if output_format == "geopackage" else ".shp"
            driver = "GPKG" if output_format == "geopackage" else "ESRI Shapefile"

            path_seg = f"{base_output_name}_segmentation{ext}"
            path_bbox = f"{base_output_name}_bboxes{ext}"

            gdf_segmentation.to_file(path_seg, driver=driver)
            gdf_bboxes.to_file(path_bbox, driver=driver)

            print(f"Resultados de segmentação salvos em: {path_seg}")
            print(f"Resultados de bounding boxes salvos em: {path_bbox}")

def main():
    # --- Configurações --- #
    MODEL_PATH = r"C:\YOLO_GABRIEL\runs\segment\experiment_8\weights\best.pt"
    GEOTIFF_PATH = r"C:\YOLO_Gabriel\imagens_uteis\SF-23-Y-B-V-2-SE-D_ORTO_RGB.tif" # Imagem RGB a ser inferida
    CHM_FOLDER = r"C:\YOLO_GABRIEL\produtos_LiDAR\3_chm" # Pasta contendo os quadrantes CHM
    OUTPUT_PATH = r"C:\YOLO_GABRIEL\predictions\prediction_output.gpkg"
    DEFAULT_CHM_CRS = 31983 # CRS padrão para CHMs sem CRS definido

    inference = YOLOGeoTIFFInference(
        model_path=MODEL_PATH,
        chm_folder=CHM_FOLDER,
        default_chm_crs=DEFAULT_CHM_CRS,
        imgsz=640, # Tamanho do patch para inferência (deve ser o mesmo do treinamento)
        overlap_ratio=0.8,
        confidence_threshold=0.8 # Limiar de confiança para detecções
    )

    inference.process(
        geotiff_path=GEOTIFF_PATH,
        output_path=OUTPUT_PATH,
        min_area_pixels=10, # Área mínima em pixels para um polígono ser salvo
        output_format="geopackage"
    )

if __name__ == '__main__':
    main()
