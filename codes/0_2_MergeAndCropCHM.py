import os
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from shapely.geometry import box, Polygon, MultiPolygon
from tqdm import tqdm

def merge_and_crop_chm(
    input_chm_dir: str,
    bbox_shapefile_path: str,
    output_crop_dir: str,
    target_crs_epsg: int
):
    """
    1. Faz o mosaico de todos os rasters CHM na pasta de entrada.
    2. Reprojetar o mosaico para o CRS alvo.
    3. Recorta o mosaico CHM reprojetado usando as geometrias do shapefile.
    4. Salva os recortes na pasta de saída, aplicando a regra de 100% de contenção.
    """
    os.makedirs(output_crop_dir, exist_ok=True)

    # 1. Carregar o shapefile de bounding boxes
    print(f"Carregando shapefile de bounding boxes: {bbox_shapefile_path}")
    try:
        bboxes_gdf = gpd.read_file(bbox_shapefile_path)
        if bboxes_gdf.crs is None:
            raise ValueError("Shapefile de bounding boxes não possui CRS definido.")
    except Exception as e:
        print(f"Erro ao carregar o shapefile {bbox_shapefile_path}: {e}")
        return

    if bboxes_gdf.empty:
        print("O shapefile de bounding boxes está vazio. Nenhuma operação de recorte será realizada.")
        return

    target_crs = rasterio.crs.CRS.from_epsg(target_crs_epsg)
    if bboxes_gdf.crs != target_crs:
        print(f"Reprojetando bounding boxes de {bboxes_gdf.crs} para o CRS alvo {target_crs}")
        bboxes_gdf = bboxes_gdf.to_crs(target_crs)

    # 2. Fazer o mosaico de todos os CHMs
    print(f"Buscando arquivos CHM em {input_chm_dir} para mosaico...")
    chm_files = [os.path.join(input_chm_dir, f) 
                 for f in os.listdir(input_chm_dir) 
                 if f.lower().endswith((".tif", ".tiff"))]

    if not chm_files:
        print(f"Nenhum arquivo CHM válido encontrado em {input_chm_dir}")
        return

    src_chms = []
    for f in chm_files:
        try:
            src = rasterio.open(f)
            src_chms.append(src)
        except Exception as e:
            print(f"Aviso: Não foi possível abrir o raster CHM {os.path.basename(f)}: {e}. Ignorando.")

    if not src_chms:
        print("Nenhum raster CHM válido para mosaico.")
        return

    print(f"Criando mosaico a partir de {len(src_chms)} arquivos CHM...")
    mosaic, out_transform = merge(src_chms)

    # Atualizar metadados do mosaico
    out_meta = src_chms[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": src_chms[0].crs,
        "nodata": -9999
    })

    # Calcular os limites geográficos do mosaico para a reprojeção
    mosaic_bounds = array_bounds(out_meta["height"], out_meta["width"], out_meta["transform"])

    for s in src_chms: s.close()

    # 3. Reprojetar o mosaico para o CRS alvo (EPSG:31983)
    print(f"Reprojetando mosaico CHM de {out_meta['crs']} para o CRS alvo {target_crs}...")
    transform, width, height = calculate_default_transform(
        out_meta["crs"], target_crs, out_meta["width"], out_meta["height"], *mosaic_bounds
    )

    # Criar um MemoryFile para o mosaico original para garantir que o CRS seja reconhecido
    with rasterio.io.MemoryFile() as memfile_src:
        with memfile_src.open(**out_meta) as src_mosaic:
            src_mosaic.write(mosaic)
            
            # Criar um MemoryFile para o mosaico reprojetado
            reprojected_meta = out_meta.copy()
            reprojected_meta.update({
                "transform": transform,
                "width": width,
                "height": height,
                "crs": target_crs
            })
            
            with rasterio.io.MemoryFile() as memfile_dst:
                with memfile_dst.open(**reprojected_meta) as dst_mosaic:
                    # Reprojeção de dataset para dataset (forma mais segura)
                    reproject(
                        source=rasterio.band(src_mosaic, 1),
                        destination=rasterio.band(dst_mosaic, 1),
                        src_transform=src_mosaic.transform,
                        src_crs=src_mosaic.crs,
                        dst_transform=dst_mosaic.transform,
                        dst_crs=dst_mosaic.crs,
                        resampling=Resampling.nearest,
                        num_threads=os.cpu_count()
                    )
                    
                    # 4. Recortar o mosaico CHM reprojetado
                    print(f"Iniciando recorte do mosaico CHM com {len(bboxes_gdf)} bounding boxes...")
                    cropped_count = 0
                    mosaic_bounds_geom = box(*dst_mosaic.bounds)

                    for idx, row in tqdm(bboxes_gdf.iterrows(), total=len(bboxes_gdf), desc="Recortando CHM"):
                        geom = row.geometry

                        if not geom.is_valid or not isinstance(geom, (Polygon, MultiPolygon)):
                            print(f"  Ignorando geometria inválida ou não-poligonal no índice {idx}.")
                            continue

                        # --- REGRA: Polígono 100% contido no mosaico CHM ---
                        if not mosaic_bounds_geom.contains(geom):
                            continue
                        # --------------------------------------------------

                        try:
                            out_image, out_transform = mask(dst_mosaic, [geom], crop=True, nodata=reprojected_meta["nodata"])
                            
                            if out_image.size == 0 or np.all(out_image == reprojected_meta["nodata"]):
                                continue

                            output_filename = f"chm_crop_bbox{idx}.tif"
                            output_path = os.path.join(output_crop_dir, output_filename)

                            out_meta_crop = reprojected_meta.copy()
                            out_meta_crop.update({
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform,
                                "count": 1
                            })

                            with rasterio.open(output_path, "w", **out_meta_crop) as dest:
                                dest.write(out_image)
                            cropped_count += 1

                        except Exception as e:
                            print(f"  Erro ao recortar bbox {idx}: {e}")
            
            if cropped_count == 0:
                print("\nAviso: Nenhum recorte CHM foi gerado. Verifique os caminhos, sobreposição e contenção.")
            else:
                print(f"\nProcessamento de recorte CHM concluído. {cropped_count} arquivos gerados em {output_crop_dir}")


if __name__ == "__main__":
    INPUT_CHM_FOLDER = r"C:\araucaria_yolo\produtos_LiDAR\3_chm"
    BBOX_SHAPEFILE = r"C:\araucaria_yolo\mascaras\bbox_mascaras.shp"
    OUTPUT_CROP_FOLDER = r"C:\araucaria_yolo\produtos_LiDAR_CHM_crop"
    TARGET_CRS_EPSG = 31983

    merge_and_crop_chm(
        INPUT_CHM_FOLDER,
        BBOX_SHAPEFILE,
        OUTPUT_CROP_FOLDER,
        TARGET_CRS_EPSG
    )
