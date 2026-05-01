import os
import cv2
import random
import numpy as np
import rasterio
import geopandas as gpd # Adicionado para carregar labels e obter species_id

# Caminho para o shapefile de labels (o mesmo usado em 1_1_GenerateDataForYOLO_CHM_v2.py)
LABELS_SHAPEFILE = r"C:\araucaria_yolo\mascaras\mascaras_merge_bbox.shp"

# Carregar labels para obter o mapeamento species_id
try:
    labels_gdf_for_species = gpd.read_file(LABELS_SHAPEFILE)
    species = sorted(labels_gdf_for_species["tree_name"].unique().tolist())
    species_id_map = {i: specie for i, specie in enumerate(species)} # Mapeamento ID -> Nome
except Exception as e:
    print(f"Erro ao carregar o shapefile de labels {LABELS_SHAPEFILE} para species_id: {e}")
    species_id_map = {} # Fallback para evitar erros

# Paths to images and labels
images_folder = r'C:\araucaria_yolo\datasets\YOLO_NIR\images\train'
labels_folder = r'C:\araucaria_yolo\datasets\YOLO_NIR\labels\train'

# Get list of image files (now looking for .tif files)
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.tif', '.tiff'))]

# Randomly select 5 images
random.seed(12500)  # Ensures reproducibility
selected_images = random.sample(image_files, min(5, len(image_files)))

# Function to draw YOLO masks in red on the image
def draw_yolo_mask(image, label_path, color=(0, 0, 255), thickness=2):
    height, width, _ = image.shape
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return image  # Return original image if label file is missing
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip invalid lines

        points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

        # Convert YOLO format (relative) to absolute pixel coordinates
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(int)

        # Draw the polygon
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

        # Get class name and display it
        class_id = int(parts[0])
        class_name = species_id_map.get(class_id, f"Class {class_id}")
        
        # Calculate text position (top-left of the bounding box of the polygon)
        x_min, y_min = np.min(points, axis=0)
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return image

# List to store image pairs (original + masked)
image_pairs = []

if not selected_images:
    print("Nenhuma imagem selecionada para visualização. Verifique se há arquivos .tif na pasta de imagens.")
else:
    for img_file in selected_images:
        img_path = os.path.join(images_folder, img_file)
        label_path = os.path.join(labels_folder, img_file.replace('.tif', '.txt').replace('.tiff', '.txt'))

        try:
            # Load image using rasterio for GeoTIFF
            with rasterio.open(img_path) as src:
                # Read bands (CHM, G, B)
                composed_data = src.read()
                
                # Convert to BGR format for OpenCV (B, G, CHM)
                # Assuming composed_data is (bands, height, width)
                # We want (height, width, BGR)
                image_bgr = np.stack((composed_data[2], composed_data[1], composed_data[0]), axis=-1)
                image_bgr = image_bgr.astype(np.uint8) # Ensure it's uint8 for OpenCV

            # Create a copy and overlay red masks
            masked_image_bgr = draw_yolo_mask(image_bgr.copy(), label_path, color=(0, 0, 255))

            # Resize to a fixed height for better visualization
            fixed_height = 200  # Set desired height
            scale = fixed_height / image_bgr.shape[0]
            new_width = int(image_bgr.shape[1] * scale)

            image_resized = cv2.resize(image_bgr, (new_width, fixed_height))
            masked_image_resized = cv2.resize(masked_image_bgr, (new_width, fixed_height))

            # Stack images side by side
            combined = np.hstack((image_resized, masked_image_resized))
            image_pairs.append(combined)
        except Exception as e:
            print(f"Erro ao processar {img_file}: {e}")

# Stack all image pairs vertically if any were processed
if image_pairs:
    final_display = np.vstack(image_pairs)
    
    # Show the final combined image
    cv2.imshow("Original (Left) vs Masked (Right)", final_display)
    
    # Wait for a key press before closing
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhum par de imagem/máscara foi processado com sucesso para exibição.")
