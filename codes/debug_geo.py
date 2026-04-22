import os
import geopandas as gpd
import rasterio
from shapely.geometry import box
import pandas as pd
import shutil
from tqdm import tqdm

def debug_full_dataset():
    data_path = r'D:\YOLO_Gabriel\imagens'
    labels_path = r"D:\YOLO_Gabriel\mascara\mascaras_merge_v2.shp"
    
    print("=== INICIANDO DEBUG GLOBAL ===")
    
    # 1. Carregar Shapefile
    try:
        mask_gdf = gpd.read_file(labels_path)
        print(f"[*] Shapefile carregado: {len(mask_gdf)} geometrias.")
    except Exception as e:
        print(f"[ERRO] Falha ao ler shapefile: {e}")
        return

    tiff_files = [f for f in os.listdir(data_path) if f.lower().endswith('.tif')]
    
    if not tiff_files:
        print(f"[ERRO] Nenhuma imagem .tif encontrada em: {data_path}")
        return

    relatorio = []

    for tiff_file in tiff_files:
        full_path = os.path.join(data_path, tiff_file)
        
        with rasterio.open(full_path) as src:
            img_bounds = src.bounds
            img_box = box(*img_bounds)
            img_crs = src.crs
            
            # Reprojetar máscara temporariamente para o CRS da imagem atual
            mask_reprojected = mask_gdf.to_crs(img_crs)
            
            # Verificar quantos polígonos intersectam
            intersectam = mask_reprojected.intersects(img_box).sum()
            
            # Verificar se há valores NoData ou se a imagem está "vazia"
            # Lemos uma pequena amostra do centro para ver se há dados
            sample = src.read(1, window=rasterio.windows.Window(src.width//2, src.height//2, 10, 10))
            has_data = "Sim" if sample.max() > 0 else "Apenas Zeros/NoData na amostra central"

            print(f"\n[Arquivo: {tiff_file}]")
            print(f"  - Resolução: {src.width}x{src.height} pixels")
            print(f"  - CRS Imagem: {img_crs}")
            print(f"  - Polígonos que tocam a imagem: {intersectam}")
            print(f"  - Amostra de dados: {has_data}")
            
            relatorio.append({
                'arquivo': tiff_file,
                'poligonos': intersectam,
                'crs_igual': (mask_gdf.crs == img_crs),
                'tem_dados': has_data
            })

    print("\n=== RESUMO FINAL ===")
    df = pd.DataFrame(relatorio)
    print(df)

    if df['poligonos'].sum() == 0:
        print("\n[ALERTA CRÍTICO] Nenhuma imagem possui polígonos sobrepostos!")
        print("Causa provável: O Shapefile e as Imagens estão em locais diferentes do mundo.")
        print(f"Dica: Verifique se um está em WGS84 (Graus) e outro em UTM (Metros).")
    elif (df['poligonos'] > 0).any():
        total_ok = df[df['poligonos'] > 0]['arquivo'].count()
        print(f"\n[OK] {total_ok} imagens possuem polígonos. Se o recorte falha, o erro é na lógica do Loop de Janelas.")

if __name__ == '__main__':
    debug_full_dataset()




def filtrar_imagens_uteis():
    # --- CONFIGURAÇÃO DE CAMINHOS ---
    path_original = r'D:\YOLO_Gabriel\imagens'
    path_destino  = r'D:\YOLO_Gabriel\imagens_uteis' # Nova pasta
    path_labels   = r"D:\YOLO_Gabriel\mascara\mascaras_merge_v2.shp"
    
    os.makedirs(path_destino, exist_ok=True)

    # 1. Carregar o Shapefile
    print("Carregando shapefile...")
    mask_gdf = gpd.read_file(path_labels)
    
    # 2. Listar arquivos TIFF
    tiff_files = [f for f in os.listdir(path_original) if f.lower().endswith('.tif')]
    print(f"Encontrados {len(tiff_files)} arquivos para analisar.")

    imagens_copiadas = 0

    # 3. Loop de Verificação
    for tiff_file in tqdm(tiff_files, desc="Filtrando imagens"):
        img_path = os.path.join(path_original, tiff_file)
        
        with rasterio.open(img_path) as src:
            # Criar a caixa delimitadora da imagem
            img_bounds = src.bounds
            img_box = box(*img_bounds)
            
            # Garantir que o CRS da máscara é o mesmo da imagem para a verificação
            # Usamos uma amostra rápida para não pesar a memória
            if mask_gdf.crs != src.crs:
                mask_temp = mask_gdf.to_crs(src.crs)
            else:
                mask_temp = mask_gdf

            # Verificar se algum polígono intersecta o retângulo da imagem
            if mask_temp.intersects(img_box).any():
                # Copiar o arquivo para a nova pasta
                shutil.copy2(img_path, os.path.join(path_destino, tiff_file))
                imagens_copiadas += 1

    print(f"\n=== PROCESSO CONCLUÍDO ===")
    print(f"Imagens analisadas: {len(tiff_files)}")
    print(f"Imagens com interseção copiadas para: {path_destino}")
    print(f"Total de imagens úteis: {imagens_copiadas}")

if __name__ == '__main__':
    filtrar_imagens_uteis()