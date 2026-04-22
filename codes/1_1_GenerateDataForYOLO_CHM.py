# -*- coding: utf-8 -*-
import os
import geopandas as gpd
import rasterio
import cv2
import numpy as np
import random
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm
from shapely.geometry import box, Polygon

def get_merged_chm(rgb_name, chm_dir):
    """
    Localiza os quadrantes CHM (I, II, III, IV), faz o merge e 
    retorna os dados alinhados.
    """
    # Limpa o nome para bater com o padrão: SF-23-Y-B-V-2-NE-E
    root_name = rgb_name.replace("_ORTO_RGB", "")
    chm_suffixes = ["I", "II", "III", "IV"]
    valid_chm_paths = []
    
    for suffix in chm_suffixes:
        path = os.path.join(chm_dir, f"{root_name}-{suffix}_chm.tif")
        if os.path.exists(path):
            valid_chm_paths.append(path)
    
    if not valid_chm_paths:
        return None, None

    src_files = [rasterio.open(p) for p in valid_chm_paths]
    mosaic, out_trans = merge(src_files)
    
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    for src in src_files: src.close()
    return mosaic[0], out_meta

def normalize_chm(chm_array, max_height=30.0):
    """
    Normaliza os valores de altura (metros) para o intervalo 0-255.
    Valores <= 0 ficam 0, valores >= max_height ficam 255.
    """
    chm_norm = np.clip(chm_array, 0, max_height) # Remove valores negativos e limita topo
    chm_norm = (chm_norm / max_height) * 255     # Escala para 0-255
    return chm_norm.astype(np.uint8)

def create_yolo_dataset_v4(tiff_image_path, chm_dir, output_folder, mask_gdf, species_id):
    tile_size = 320
    step = tile_size // 2
    val_fraction = 0.2
    
    base_name = os.path.basename(tiff_image_path).split('.')[0]
    
    # 1. Obter e Processar CHM
    chm_raw, chm_meta = get_merged_chm(base_name, chm_dir)
    
    with rasterio.open(tiff_image_path) as src_rgb:
        # Lemos apenas as bandas 2 (G) e 3 (B) inicialmente, ou todas para garantir
        rgb_data = src_rgb.read() 
        
        if chm_raw is not None:
            # Alinhamento espacial (VRT) para garantir que CHM e RGB casem pixel a pixel
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**chm_meta) as mem_ds:
                    mem_ds.write(chm_raw, 1)
                    with WarpedVRT(mem_ds, crs=src_rgb.crs, transform=src_rgb.transform, 
                                   width=src_rgb.width, height=src_rgb.height) as vrt:
                        chm_aligned = vrt.read(1)
            
            # Normalização para 8-bit
            chm_8bit = normalize_chm(chm_aligned, max_height=35.0) # Ajuste a altura máxima conforme sua floresta
            
            # Substituição: Banda 1 (Red) agora é o CHM Normalizado
            rgb_data[0] = chm_8bit
        
        # --- Lógica de Máscara e Tiling ---
        if mask_gdf.crs != src_rgb.crs:
            mask_gdf = mask_gdf.to_crs(src_rgb.crs)
            
        img_bounds_geom = box(*src_rgb.bounds)
        mask_local = mask_gdf[mask_gdf.intersects(img_bounds_geom)].copy()
        
        if mask_local.empty:
            return

        for r in tqdm(range(0, src_rgb.height - tile_size, step), desc=f"Lidar-RGB: {base_name}"):
            for c in range(0, src_rgb.width - tile_size, step):
                window = Window(col_off=c, row_off=r, width=tile_size, height=tile_size)
                win_transform = src_rgb.window_transform(window)
                window_geom = box(*rasterio.windows.bounds(window, src_rgb.transform))
                
                possible_shapes = mask_local[mask_local.intersects(window_geom)]
                if possible_shapes.empty: continue

                buffer = ""
                contem_objeto_util = False 

                for _, row in possible_shapes.iterrows():
                    full_geom = row['geometry']
                                       # Get the intersection of the polygon with the tile
                    clipped_geom = full_geom.intersection(window_geom)
                    
                    # Modificado: Incluir polígonos que são parcialmente cortados pelo tile.
                    # Isso é essencial para a classe "sombra", que costuma ser maior que o tile.
                    if clipped_geom.is_empty:
                        continue
                    
                    contem_objeto_util = True

                    # Conversão YOLO (simplificada para o exemplo)
                    # Garante que estamos lidando apenas com Polygons
                    if isinstance(clipped_geom, Polygon):
                        polys = [clipped_geom]
                    elif clipped_geom.geom_type == 'MultiPolygon':
                        polys = [p for p in clipped_geom.geoms if isinstance(p, Polygon)]
                    else:
                        continue # Ignora outras geometrias como LineString ou Point
                    for poly in polys:
                        coords = list(poly.exterior.coords)
                        yolo_coords = []
                        for x_c, y_c in coords:
                            row_p, col_p = rasterio.transform.rowcol(win_transform, x_c, y_c)
                            yolo_coords.append(f"{np.clip(col_p/tile_size,0,1):.6f} {np.clip(row_p/tile_size,0,1):.6f}")
                        buffer += f"{species_id[row['tree_name']]} " + " ".join(yolo_coords) + "\n"

                    
                if contem_objeto_util and buffer.strip() != "":
                    # Extração do Tile do array modificado
                    tile = rgb_data[:, r:r+tile_size, c:c+tile_size]
                    if np.mean(tile == 0) > 0.8: continue

                    # Reordena canais para (H, W, C)
                    tile = np.moveaxis(tile, 0, -1)[:, :, :3]
                    
                    # Converte para BGR para o OpenCV salvar corretamente
                    # Original RGB -> OpenCV BGR. Como trocamos R pelo CHM:
                    # [CHM, G, B] -> cv2.cvtColor -> [B, G, CHM]
                    tile_bgr = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2BGR)

                    prefix = 'val' if random.random() < val_fraction else 'train'
                    file_id = f"{base_name}_r{r}_c{c}"
                    
                    # Salvamento
                    out_img_dir = os.path.join(output_folder, 'images', prefix)
                    out_lab_dir = os.path.join(output_folder, 'labels', prefix)
                    os.makedirs(out_img_dir, exist_ok=True)
                    os.makedirs(out_lab_dir, exist_ok=True)
                    
                    cv2.imwrite(os.path.join(out_img_dir, f"{file_id}.jpg"), tile_bgr)
                    with open(os.path.join(out_lab_dir, f"{file_id}.txt"), "w") as f:
                        f.write(buffer)

# No main(), lembre-se de passar o caminho do CHM:
# create_yolo_dataset_v4(tiff_path, Data_CHM, output_folder, mask_gdf, species_id)
def main():
    
    Data_RGB = r'C:\YOLO_GABRIEL\imagens_uteis'
    Data_CHM = r'C:\YOLO_GABRIEL\produtos_LiDAR\3_chm'
    labels   = r"C:\YOLO_Gabriel\mascara\mascaras_total.shp"
    
    mask_gdf = gpd.read_file(labels)
    mask_gdf['geometry'] = mask_gdf['geometry'].make_valid()
    mask_gdf = mask_gdf[mask_gdf.geometry.notnull() & ~mask_gdf.geometry.is_empty]

    species = sorted(mask_gdf['tree_name'].unique().tolist())
    species_id = {specie: i for i, specie in enumerate(species)}

    tiff_files = [f for f in os.listdir(Data_RGB) if f.lower().endswith('.tif')]

    for i, tiff_file in enumerate(tiff_files, start=1):
        tiff_path = os.path.join(Data_RGB, tiff_file)
        output_folder = f"C:/YOLO_Gabriel/datasets/YOLO_CHM_Composite/"

        print(f"\nProcessando {i}/{len(tiff_files)}: {tiff_file}")
        create_yolo_dataset_v4(tiff_path, Data_CHM, output_folder, mask_gdf, species_id)

if __name__ == '__main__':
    main()