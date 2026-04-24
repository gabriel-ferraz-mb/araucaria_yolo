import os
import geopandas as gpd
import rasterio
import numpy as np
import random
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
from shapely.geometry import box, Polygon, MultiPolygon
import re

def normalize_chm(chm_array, max_height=30.0):
    """
    Normaliza os valores de altura (metros) para o intervalo 0-255.
    Valores <= 0 ficam 0, valores >= max_height ficam 255.
    """
    chm_norm = np.clip(chm_array, 0, max_height)
    chm_norm = (chm_norm / max_height) * 255
    return chm_norm.astype(np.uint8)

def generate_yolo_dataset_from_cropped_rasters(
    input_rgb_crop_dir: str,
    input_chm_crop_dir: str,
    labels_shapefile_path: str,
    output_folder: str,
    species_id: dict
):
    """
    Gera o dataset YOLO a partir dos arquivos RGB e CHM já recortados.
    Composição de 3 bandas (CHM, G, B) com reamostragem automática do CHM.
    """
    tile_size = 640 # Alterado para 640
    step = tile_size // 2 # Ajustado para 320
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
    
    # Mapear CHM files por um identificador comum (ex: bbox123)
    all_chm_crop_files_map = {}
    for f in os.listdir(input_chm_crop_dir):
        if f.lower().endswith((".tif", ".tiff")):
            base_name = os.path.splitext(f)[0]
            match = re.search(r'bbox(\d+)', base_name)
            if match:
                all_chm_crop_files_map[match.group(1)] = os.path.join(input_chm_crop_dir, f)

    if not all_rgb_crop_files:
        print(f"Nenhum arquivo RGB recortado válido encontrado em {input_rgb_crop_dir}")
        return

    total_yolo_patches_generated = 0
    print(f"Iniciando geração de patches YOLO a partir de {len(all_rgb_crop_files)} RGBs recortados...")

    for rgb_crop_filename in tqdm(all_rgb_crop_files, desc="Processando RGBs recortados para YOLO"):
        rgb_crop_path = os.path.join(input_rgb_crop_dir, rgb_crop_filename)
        rgb_crop_base_name = os.path.splitext(rgb_crop_filename)[0]

        match = re.search(r'_crop_(\d+)', rgb_crop_base_name)
        if not match:
            print(f"  Aviso: Não foi possível extrair o ID do crop do nome do arquivo RGB: {rgb_crop_filename}. Pulando.")
            continue
        crop_id_key = match.group(1)

        chm_crop_path = all_chm_crop_files_map.get(crop_id_key)

        if not chm_crop_path:
            print(f"  Aviso: CHM recortado correspondente a {rgb_crop_filename} (com id {crop_id_key}) não encontrado. Pulando.")
            continue

        try:
            with rasterio.open(rgb_crop_path) as src_rgb_crop:
                with rasterio.open(chm_crop_path) as src_chm_crop:

                    # Verificar se os rasters têm o mesmo CRS
                    if src_rgb_crop.crs != src_chm_crop.crs:
                        print(f"  Erro: CRS de RGB ({src_rgb_crop.crs}) e CHM ({src_chm_crop.crs}) recortados são diferentes para {rgb_crop_filename}. Pulando.")
                        continue
                    
                    # Reamostrar CHM para as dimensões do RGB se forem diferentes
                    if src_rgb_crop.shape != src_chm_crop.shape:
                        # print(f"  Reamostrando CHM para as dimensões do RGB: {src_rgb_crop.shape}")
                        chm_data = np.empty((1, src_rgb_crop.height, src_rgb_crop.width), dtype=src_chm_crop.dtypes[0])
                        reproject(
                            source=rasterio.band(src_chm_crop, 1),
                            destination=chm_data,
                            src_transform=src_chm_crop.transform,
                            src_crs=src_chm_crop.crs,
                            dst_transform=src_rgb_crop.transform,
                            dst_crs=src_rgb_crop.crs,
                            resampling=Resampling.bilinear
                        )
                        chm_data = chm_data[0] # Remover dimensão da banda
                    else:
                        chm_data = src_chm_crop.read(1)

                    # Ler dados RGB
                    rgb_data = src_rgb_crop.read()

                    # Normalizar CHM
                    chm_8bit = normalize_chm(chm_data, max_height=35.0)

                    # Compor o raster de 3 bandas (CHM, G, B)
                    composed_data = np.stack((chm_8bit, rgb_data[1], rgb_data[2]), axis=0)

                    # Reprojetar labels para o CRS do insumo
                    insumo_crs = src_rgb_crop.crs
                    if labels_gdf.crs and labels_gdf.crs != insumo_crs:
                        labels_gdf_proj = labels_gdf.to_crs(insumo_crs)
                    else:
                        labels_gdf_proj = labels_gdf

                    # Lógica de Tiling e Geração de Patches YOLO
                    for r_tile in range(0, composed_data.shape[1] - tile_size + 1, step):
                        for c_tile in range(0, composed_data.shape[2] - tile_size + 1, step):
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
                                    # Ignorar outros tipos de geometria que não sejam polígonos
                                    continue
                                
                                for poly in polys_to_process:
                                    # Garantir que o polígono é válido após a interseção
                                    if not poly.is_valid:
                                        poly = poly.buffer(0) # Tentar corrigir geometria inválida
                                    
                                    # Se o buffer(0) resultar em uma geometria vazia ou inválida, pular
                                    if poly.is_empty or not poly.is_valid:
                                        continue

                                    coords = list(poly.exterior.coords) # Coordenadas do polígono intersectado
                                    yolo_coords = []
                                    for x_c, y_c in coords:
                                        row_p, col_p = rasterio.transform.rowcol(win_transform, x_c, y_c)
                                        # Clipar as coordenadas para garantir que fiquem entre 0 e 1
                                        yolo_coords.append(f"{np.clip(col_p/tile_size,0,1):.6f} {np.clip(row_p/tile_size,0,1):.6f}")
                                    buffer += f"{species_id[label_row['tree_name']]} " + " ".join(yolo_coords) + "\n"

                            if contem_objeto_util and buffer.strip() != "":
                                tile_data = composed_data[:, r_tile:r_tile+tile_size, c_tile:c_tile+tile_size]
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
                                    'dtype': composed_data.dtype
                                })
                                output_tiff_path = os.path.join(out_img_dir, f"{file_id}.tif")
                                with rasterio.open(output_tiff_path, 'w', **tile_meta) as dst:
                                    dst.write(tile_data)

                                with open(os.path.join(out_lab_dir, f"{file_id}.txt"), "w") as f:
                                    f.write(buffer)
                                total_yolo_patches_generated += 1

        except Exception as e:
            print(f"  Erro ao processar par RGB/CHM recortado {rgb_crop_filename}: {e}")

    if total_yolo_patches_generated == 0:
        print("\nAviso: Nenhum patch YOLO foi gerado. Verifique os insumos recortados, labels e regras de contenção.")
    else:
        print(f"\nProcessamento de dataset YOLO concluído. {total_yolo_patches_generated} patches gerados.")

def main():
    
    INPUT_RGB_CROP_FOLDER = r'C:\araucaria_yolo\imagens_uteis_crop'
    INPUT_CHM_CROP_FOLDER = r'C:\araucaria_yolo\produtos_LiDAR_CHM_crop'
    LABELS_SHAPEFILE = r"C:\araucaria_yolo\mascaras\mascaras_merge_bbox.shp"

    OUTPUT_YOLO_DATASET_FOLDER = r"C:/araucaria_yolo/datasets/YOLO_CHM_Composite/"
    
    print(f"Carregando shapefile de labels para species_id: {LABELS_SHAPEFILE}")
    try:
        labels_gdf_for_species = gpd.read_file(LABELS_SHAPEFILE)
    except Exception as e:
        print(f"Erro ao carregar o shapefile de labels {LABELS_SHAPEFILE}: {e}")
        return

    species = sorted(labels_gdf_for_species['tree_name'].unique().tolist())
    species_id = {specie: i for i, specie in enumerate(species)}

    print("\n--- GERANDO DATASET YOLO A PARTIR DOS INSUMOS RECORTADOS E LABELS ---")
    generate_yolo_dataset_from_cropped_rasters(
        INPUT_RGB_CROP_FOLDER,
        INPUT_CHM_CROP_FOLDER,
        LABELS_SHAPEFILE,
        OUTPUT_YOLO_DATASET_FOLDER,
        species_id
    )

if __name__ == '__main__':
    main()
