import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from tqdm import tqdm

def crop_rasters_by_bbox(
    input_raster_dir: str,
    bbox_shapefile_path: str,
    output_crop_dir: str,
    name_suffix: str = "",
    default_raster_crs=None # Novo parâmetro para CRS padrão se o raster não tiver um
):
    """
    Recorta rasters GeoTIFF de um diretório usando os polígonos de um shapefile
    e salva os recortes em um novo diretório.

    Args:
        input_raster_dir (str): Caminho para o diretório contendo os rasters GeoTIFF de entrada.
        bbox_shapefile_path (str): Caminho para o arquivo shapefile (.shp) contendo os polígonos de recorte.
        output_crop_dir (str): Caminho para o diretório onde os rasters recortados serão salvos.
        name_suffix (str): Sufixo opcional para adicionar ao nome do arquivo de saída (ex: ".chm" para CHM).
        default_raster_crs: CRS a ser assumido para o raster se ele não tiver um CRS definido.
                            Pode ser um código EPSG (int) ou uma string WKT/PROJ4.
    """

    os.makedirs(output_crop_dir, exist_ok=True)

    print(f"Carregando shapefile de bounding boxes: {bbox_shapefile_path}")
    try:
        bboxes_gdf = gpd.read_file(bbox_shapefile_path)
    except Exception as e:
        print(f"Erro ao carregar o shapefile {bbox_shapefile_path}: {e}")
        return

    if bboxes_gdf.empty:
        print("O shapefile de bounding boxes está vazio. Nenhuma operação de recorte será realizada.")
        return

    raster_files = [f for f in os.listdir(input_raster_dir) if f.lower().endswith((".tif", ".tiff"))]
    if not raster_files:
        print(f"Nenhum arquivo GeoTIFF encontrado em {input_raster_dir}")
        return

    print(f"Encontrados {len(raster_files)} arquivos GeoTIFF para processar.")

    for raster_filename in tqdm(raster_files, desc="Processando rasters"):
        raster_path = os.path.join(input_raster_dir, raster_filename)
        print(f"\nProcessando raster: {raster_filename}")

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

            # Lógica para lidar com raster sem CRS definido
            if raster_crs is None:
                print(f"  Aviso: Raster \'{raster_filename}\' não possui CRS definido.")
                if default_raster_crs:
                    print(f"  Assumindo CRS do raster como: {default_raster_crs}")
                    raster_crs = rasterio.crs.CRS.from_epsg(default_raster_crs) if isinstance(default_raster_crs, int) else rasterio.crs.CRS.from_string(default_raster_crs)
                else:
                    print("  Não foi fornecido um CRS padrão. Pulando reprojeção e assumindo que o raster e o shapefile já estão alinhados.")
                    bboxes_gdf_proj = bboxes_gdf # Assume que já estão no mesmo CRS
            
            if raster_crs and bboxes_gdf.crs and bboxes_gdf.crs != raster_crs:
                print(f"  Reprojetando bounding boxes de {bboxes_gdf.crs} para {raster_crs}")
                bboxes_gdf_proj = bboxes_gdf.to_crs(raster_crs)
            else:
                bboxes_gdf_proj = bboxes_gdf

            # Criar a geometria dos limites do raster para a verificação de contenção
            raster_bounds_geom = box(*src.bounds)

            for idx, row in bboxes_gdf_proj.iterrows():
                geom = row.geometry

                if not geom.is_valid or not isinstance(geom, (Polygon, MultiPolygon)):
                    print(f"  Ignorando geometria inválida ou não-poligonal no índice {idx}.")
                    continue

                # --- REGRA: Polígono 100% contido no raster ---
                if not raster_bounds_geom.contains(geom):
                    print(f"  Aviso: Bounding box {idx} não está 100% contida no raster {raster_filename}. Pulando recorte.")
                    continue
                # --------------------------------------------------

                try:
                    out_image, out_transform = mask(src, [geom], crop=True)

                    # Verificar se o recorte resultou em imagem vazia ou apenas nodata
                    if out_image.size == 0 or (src.nodata is not None and np.all(out_image == src.nodata)):
                        print(f"  Aviso: Recorte para bbox {idx} resultou em imagem vazia ou apenas nodata. Pulando.")
                        continue

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    # Se um CRS foi definido ou assumido, atualiza os metadados
                    if raster_crs: 
                        out_meta.update({"crs": raster_crs})

                    base_name = os.path.splitext(raster_filename)[0]
                    output_filename = f"{base_name}_crop_{idx}{name_suffix}.tif"
                    output_path = os.path.join(output_crop_dir, output_filename)

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(out_image)

                except Exception as e:
                    print(f"  Erro ao recortar o raster {raster_filename} com a geometria {idx}: {e}")

    print("\nProcessamento de recorte de rasters concluído.")

if __name__ == "__main__":
    # --- Configurações para RGB --- #
    INPUT_RASTER_FOLDER_RGB = r"C:\araucaria_yolo\imagens_uteis"
    BBOX_SHAPEFILE = r"C:\araucaria_yolo\mascaras\bbox_mascaras.shp"
    OUTPUT_CROP_FOLDER_RGB = r"C:\araucaria_yolo\imagens_uteis_crop"
    
    print("\n--- Recortando Rasters RGB ---")
    crop_rasters_by_bbox(INPUT_RASTER_FOLDER_RGB, BBOX_SHAPEFILE, OUTPUT_CROP_FOLDER_RGB)
