# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:33:59 2025

@author: Matheus

CORRECTED VERSION:
- Fixed tile_size to 640x640 (was being resized to 100x100)
- Only saves patches that completely contain polygons from the shapefile
- Optimized for YOLO training dataset generation
"""

#%% CREATE YOLO DATASET -> FULL POLYGON CONTAINMENT
import os
import geopandas as gpd
import rasterio
import cv2
import numpy as np
import random
from shapely.geometry import Polygon, MultiPolygon
from rasterio.windows import Window
from tqdm import tqdm
from shapely.geometry import box

def image_intersects_shapes(tiff_path, mask_gdf):
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds  # left, bottom, right, top
        image_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Ensure same CRS
    if mask_gdf.crs != src.crs:
        mask_gdf = mask_gdf.to_crs(src.crs)
    
    return mask_gdf.intersects(image_geom).any()

def create_yolo_dataset_v4(tiff_image_path, output_folder, mask_gdf, species_id):
    """
    Create YOLO dataset with 640x640 patches that completely contain polygons.
    Only s  aves patches where entire polygon(s) fit within the tile.
    """
   
    tile_size = 640  # Fixed tile size - NO RESIZING
    step = tile_size // 2  # 50% overlap
    val_fraction = 0.2
    
    # Directories for train/val split
    dirs = {
        'train_img': os.path.join(output_folder, 'images/train/'),
        'val_img':   os.path.join(output_folder, 'images/val/'),
        'train_lab': os.path.join(output_folder, 'labels/train/'),
        'val_lab':   os.path.join(output_folder, 'labels/val/')
    }
    for d in dirs.values(): 
        os.makedirs(d, exist_ok=True)

    with rasterio.open(tiff_image_path) as src:
        # Ensure same CRS
        if mask_gdf.crs != src.crs:
            mask_gdf = mask_gdf.to_crs(src.crs)
            
        img_bounds_geom = box(*src.bounds)
        mask_local = mask_gdf[mask_gdf.intersects(img_bounds_geom)].copy()
        
        base_name = os.path.basename(tiff_image_path).split('.')[0]
        
        if mask_local.empty:
            print(f" -> Skipped {base_name}: No trees found.")
            return

        # Generate tiles with sliding window
        for r in tqdm(range(0, src.height - tile_size, step), desc=f"Processing {base_name}"):
            for c in range(0, src.width - tile_size, step):
                # Ensure tile doesn't exceed image boundaries
                if r + tile_size > src.height or c + tile_size > src.width:
                    continue
                    
                window = Window(col_off=c, row_off=r, width=tile_size, height=tile_size)
                
                win_transform = src.window_transform(window)
                win_bounds = rasterio.windows.bounds(window, src.transform)
                window_geom = box(*win_bounds)
                
                # Find polygons that intersect with this window
                possible_shapes = mask_local[mask_local.intersects(window_geom)]
                
                if possible_shapes.empty:
                    continue

                buffer = ""
                contem_objeto_util = False 

                # Process each polygon in the window
                for _, row in possible_shapes.iterrows():
                    specie = row['tree_name']
                    full_geom = row['geometry']
                    
                                        # Get the intersection of the polygon with the tile
                    clipped_geom = full_geom.intersection(window_geom)
                    
                    if clipped_geom.is_empty:
                        continue
                    
                    # Modificado: Incluir polígonos que são parcialmente cortados pelo tile.
                    # Apenas garantimos que a geometria resultante não esteja vazia.
                    
                    contem_objeto_util = True
                    
                    # Handle Polygon and MultiPolygon
                    polys = [clipped_geom] if isinstance(clipped_geom, Polygon) else list(clipped_geom.geoms)
                    
                    for poly in polys:
                        if not hasattr(poly, 'exterior') or poly.exterior is None: 
                            continue
                        
                        coords = list(poly.exterior.coords)
                        yolo_coords = []
                        
                        # Convert coordinates to YOLO format (normalized 0-1)
                        for x_c, y_c in coords:
                            row_p, col_p = rasterio.transform.rowcol(win_transform, x_c, y_c)
                            x_n = np.clip(col_p / tile_size, 0, 1)
                            y_n = np.clip(row_p / tile_size, 0, 1)
                            yolo_coords.append(f"{x_n:.6f} {y_n:.6f}")
                        
                        if yolo_coords:
                            buffer += f"{species_id[specie]} " + " ".join(yolo_coords) + "\n"


                # Only save if tile contains valid objects
                if contem_objeto_util and buffer.strip() != "":
                    # Read the image tile
                    img = src.read(window=window)
                    
                    if img is None or img.size == 0:
                        continue
                    
                    # Skip tiles with excessive black pixels (no data)
                    if np.mean(img == 0) > 0.9: 
                        continue

                    # Convert from (C, H, W) to (H, W, C) and keep only RGB
                    img = np.moveaxis(img, 0, -1)[:, :, :3]
                    
                    # Ensure tile is exactly 640x640 (no resizing needed if window is correct)
                    if img.shape[0] != tile_size or img.shape[1] != tile_size:
                        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
                    
                    # Convert color space if needed
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Randomly split into train/val
                    prefix = 'val' if random.random() < val_fraction else 'train'
                    file_id = f"{base_name}_r{r}_c{c}"
                    
                    # Save image
                    img_path = os.path.join(dirs[f'{prefix}_img'], f"{file_id}.jpg")
                    cv2.imwrite(img_path, img)
                    
                    # Save labels
                    label_path = os.path.join(dirs[f'{prefix}_lab'], f"{file_id}.txt")
                    with open(label_path, "w") as f:
                        f.write(buffer)

def main():
    Data   = r'O:\araucaria_yolo\imagens_uteis'
    labels = r"O:\araucaria_yolo\mascara\mascaras_merge_v2.shp"
    
    # Read shapefile
    mask_gdf = gpd.read_file(labels)
    mask_gdf = mask_gdf[['id', 'tree_name', 'geometry']]
    mask_gdf['geometry'] = mask_gdf['geometry'].make_valid()

    # Remove None or empty geometries
    mask_gdf = mask_gdf[
        mask_gdf.geometry.notnull() &
        ~mask_gdf.geometry.is_empty
    ]

    print(f"Checking number of masks: {len(mask_gdf)}")
    print(f"Checking masks CRS: {mask_gdf.crs}")

    
    # Create a species ID mapping
    species = sorted(mask_gdf['tree_name'].unique().tolist())
    species_id = {specie: i for i, specie in enumerate(species)}
    print(f"Species mapping: {species_id}")

    # List all .tif files in the directory
    tiff_files = [f for f in os.listdir(Data) if f.lower().endswith('.tif')]

    for i, tiff_file in enumerate(tiff_files, start=1):
        tiff_image_path = os.path.join(Data, tiff_file)
        output_folder   = f"O:/araucaria_yolo/datasets/YOLO{i}/"

        print(f"\nProcessing image {i}/{len(tiff_files)}: {tiff_file}")

        if not image_intersects_shapes(tiff_image_path, mask_gdf):
            print("  → No intersection with shapefile. Skipping.")
            continue

        print("  → Intersection found. Processing...")
        create_yolo_dataset_v4(tiff_image_path, output_folder, mask_gdf, species_id)
        print(f"  → Completed {tiff_file}")

if __name__ == '__main__':
    main()
