import os
import geopandas as gpd
import rasterio
import numpy as np
import random
from rasterio.windows import Window
from tqdm import tqdm
from shapely.geometry import box, Polygon, MultiPolygon
import re

def generate_yolo_dataset_from_cropped_rasters(
    input_rgb_crop_dir: str,
    labels_shapefile_path: str,
    output_folder: str,
    species_id: dict
):
    """
    Gera o dataset YOLO a partir dos arquivos RGB recortados.
    Usa as 3 bandas originais (R, G, B).
    """
    tile_size = 640 
    step = tile_size // 2 
    val_fraction = 0.2

    print(f"Carregando shapefile de labels para YOLO: {labels_shapefile_path}")
    try:
        labels_gdf = gpd.read_file(labels_shapefile_path)
    except Exception as e:
        print(f"Erro ao carregar o shapefile {labels_shapefile_path}: {e}")
        return

    labels_gdf["geometry"] = labels_gdf["geometry"].make_valid()
    labels_gdf = labels_gdf[labels_gdf.geometry.notnull() & ~labels_gdf.geometry.is_empty]

    if labels_gdf.empty:
        print("O shapefile de labels está vazio ou não contém geometrias válidas. Nenhuma operação será realizada.")
        return

    all_rgb_crop_files = [f for f in os.listdir(input_rgb_crop_dir) if f.lower().endswith((".tif", ".tiff"))]
    
    if not all_rgb_crop_files:
        print(f"Nenhum arquivo RGB recortado válido encontrado em {input_rgb_crop_dir}")
        return

    total_yolo_patches_generated = 0
    print(f"Iniciando geração de patches YOLO a partir de {len(all_rgb_crop_files)} RGBs recortados...")

    for rgb_crop_filename in tqdm(all_rgb_crop_files, desc="Processando RGBs recortados para YOLO"):
        rgb_crop_path = os.path.join(input_rgb_crop_dir, rgb_crop_filename)
        rgb_crop_base_name = os.path.splitext(rgb_crop_filename)[0]

        try:
            with rasterio.open(rgb_crop_path) as src_rgb_crop:
                # Ler dados RGB (3 bandas)
                rgb_data = src_rgb_crop.read()
                
                # Garantir que temos pelo menos 3 bandas
                if rgb_data.shape[0] < 3:
                    print(f"  Aviso: O arquivo {rgb_crop_filename} possui menos de 3 bandas. Pulando.")
                    continue

                # Reprojetar labels para o CRS do insumo
                insumo_crs = src_rgb_crop.crs
                if labels_gdf.crs and labels_gdf.crs != insumo_crs:
                    labels_gdf_proj = labels_gdf.to_crs(insumo_crs)
                else:
                    labels_gdf_proj = labels_gdf

                # Lógica de Tiling e Geração de Patches YOLO
                for r_tile in range(0, rgb_data.shape[1] - tile_size + 1, step):
                    for c_tile in range(0, rgb_data.shape[2] - tile_size + 1, step):
                        window = Window(col_off=c_tile, row_off=r_tile, width=tile_size, height=tile_size)
                        win_transform = src_rgb_crop.window_transform(window)
                        window_geom = box(*rasterio.windows.bounds(window, src_rgb_crop.transform))
                        
                        # Buscar labels que intersectam a janela (patch)
                        possible_labels = labels_gdf_proj[labels_gdf_proj.intersects(window_geom)]
                        if possible_labels.empty: continue

                        buffer = ""
                        contem_objeto_util = False 

                        for _, label_row in possible_labels.iterrows():
                            full_geom = label_row["geometry"]
                            
                            # Calcular a interseção da geometria do label com a janela do patch
                            intersected_geom = full_geom.intersection(window_geom)

                            if intersected_geom.is_empty:
                                continue

                            # Calcular a porcentagem de interseção
                            intersection_percentage = (intersected_geom.area / full_geom.area) * 100

                            # Verificar se a interseção é de pelo menos 70%
                            if intersection_percentage < 70.0:
                                continue

                            contem_objeto_util = True
                            polys_to_process = []

                            if isinstance(intersected_geom, Polygon):
                                polys_to_process = [intersected_geom]
                            elif intersected_geom.geom_type == 'MultiPolygon':
                                polys_to_process = [p for p in intersected_geom.geoms if isinstance(p, Polygon)]
                            else:
                                continue
                            
                            for poly in polys_to_process:
                                if not poly.is_valid:
                                    poly = poly.buffer(0)
                                
                                if poly.is_empty or not poly.is_valid:
                                    continue

                                coords = list(poly.exterior.coords)
                                yolo_coords = []
                                for x_c, y_c in coords:
                                    row_p, col_p = rasterio.transform.rowcol(win_transform, x_c, y_c)
                                    yolo_coords.append(f"{np.clip(col_p/tile_size,0,1):.6f} {np.clip(row_p/tile_size,0,1):.6f}")
                                buffer += f"{species_id[label_row['tree_name']]} " + " ".join(yolo_coords) + "\n"

                        if contem_objeto_util and buffer.strip() != "":
                            tile_data = rgb_data[:3, r_tile:r_tile+tile_size, c_tile:c_tile+tile_size]
                            # Pular patches com mais de 80% de pixels pretos (NoData)
                            if np.mean(tile_data == 0) > 0.8: continue

                            prefix = 'val' if random.random() < val_fraction else 'train'
                            file_id = f"{rgb_crop_base_name}_r{r_tile}_c{c_tile}"
                            
                            out_img_dir = os.path.join(output_folder, 'images', prefix)
                            out_lab_dir = os.path.join(output_folder, 'labels', prefix)
                            os.makedirs(out_img_dir, exist_ok=True)
                            os.makedirs(out_lab_dir, exist_ok=True)
                            
                            tile_meta = src_rgb_crop.meta.copy()
                            tile_meta.update({
                                'driver': 'GTiff',
                                'height': tile_size,
                                'width': tile_size,
                                'count': 3,
                                'transform': win_transform,
                                'dtype': rgb_data.dtype
                            })
                            output_tiff_path = os.path.join(out_img_dir, f"{file_id}.tif")
                            with rasterio.open(output_tiff_path, 'w', **tile_meta) as dst:
                                dst.write(tile_data)

                            with open(os.path.join(out_lab_dir, f"{file_id}.txt"), "w") as f:
                                f.write(buffer)
                            total_yolo_patches_generated += 1

        except Exception as e:
            print(f"  Erro ao processar RGB recortado {rgb_crop_filename}: {e}")

    if total_yolo_patches_generated == 0:
        print("\nAviso: Nenhum patch YOLO foi gerado. Verifique os insumos recortados, labels e regras de contenção.")
    else:
        print(f"\nProcessamento de dataset YOLO concluído. {total_yolo_patches_generated} patches gerados.")

def main():
    
    INPUT_RGB_CROP_FOLDER = r'C:\YOLO_Gabriel\imagens_uteis_crop'
    LABELS_SHAPEFILE = r"C:\YOLO_Gabriel\mascaras\mascaras_merge_bbox.shp"

    # Pasta de saída alterada para refletir que é apenas RGB
    OUTPUT_YOLO_DATASET_FOLDER = r"C:/YOLO_Gabriel/datasets/YOLO_RGB_Pure/"
    
    print(f"Carregando shapefile de labels para species_id: {LABELS_SHAPEFILE}")
    try:
        labels_gdf_for_species = gpd.read_file(LABELS_SHAPEFILE)
    except Exception as e:
        print(f"Erro ao carregar o shapefile de labels {LABELS_SHAPEFILE}: {e}")
        return

    species = sorted(labels_gdf_for_species['tree_name'].unique().tolist())
    species_id = {specie: i for i, specie in enumerate(species)}

    print("\n--- GERANDO DATASET YOLO A PARTIR DOS INSUMOS RGB E LABELS ---")
    generate_yolo_dataset_from_cropped_rasters(
        INPUT_RGB_CROP_FOLDER,
        LABELS_SHAPEFILE,
        OUTPUT_YOLO_DATASET_FOLDER,
        species_id
    )

if __name__ == '__main__':
    main()
