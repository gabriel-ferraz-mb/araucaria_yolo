#region MERGE YOLO DATASETS
import os
import shutil
import time
import sys

def merge_yolo_folders(source_root, target_root):
    # Paths for the target YOLO_ALL folder
    target_images_train = os.path.join(target_root, 'images/train')
    target_images_val = os.path.join(target_root, 'images/val')
    target_labels_train = os.path.join(target_root, 'labels/train')
    target_labels_val = os.path.join(target_root, 'labels/val')

    # Create the target directories if they don't exist
    os.makedirs(target_images_train, exist_ok=True)
    os.makedirs(target_images_val, exist_ok=True)
    os.makedirs(target_labels_train, exist_ok=True)
    os.makedirs(target_labels_val, exist_ok=True)

    # Get all folders within the source root that start with 'YOLO'
    yolo_folders = [f for f in os.listdir(source_root) if f.startswith('YOLO') and os.path.isdir(os.path.join(source_root, f))]

    for yolo_folder in yolo_folders:
        # Define paths for 'train' and 'val' subfolders in both images and labels
        images_train = os.path.join(source_root, yolo_folder, 'images', 'train')
        images_val = os.path.join(source_root, yolo_folder, 'images', 'val')
        labels_train = os.path.join(source_root, yolo_folder, 'labels', 'train')
        labels_val = os.path.join(source_root, yolo_folder, 'labels', 'val')

        # Copy all image files from 'train' and 'val' to the corresponding target subfolders
        if os.path.exists(images_train):
            copy_files_with_retry(images_train, target_images_train)
        if os.path.exists(images_val):
            copy_files_with_retry(images_val, target_images_val)

        # Copy all label files from 'train' and 'val' to the corresponding target subfolders
        if os.path.exists(labels_train):
            copy_files_with_retry(labels_train, target_labels_train)
        if os.path.exists(labels_val):
            copy_files_with_retry(labels_val, target_labels_val)

    print("Merging complete.")

def copy_files_with_retry(source_folder, target_folder, max_retries=5, delay=1):
    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        # Construct full file path
        source_file_path = os.path.join(source_folder, file_name)
        target_file_path = os.path.join(target_folder, file_name)

        # Check if it's a file and not a directory
        if os.path.isfile(source_file_path):
            # Retry the copy process in case of permission errors
            for attempt in range(max_retries):
                try:
                    # Copy the file to the target folder
                    shutil.copy2(source_file_path, target_file_path)
                    print(f"Copied {source_file_path} to {target_file_path}")
                    break  # If the copy is successful, exit the retry loop
                except PermissionError as e:
                    if "WinError 32" in str(e):
                        print(f"PermissionError: {e}. File: {source_file_path}. Skipping this file.")
                        break  # Skip the file if it's in use by another process
                    else:
                        print(f"PermissionError: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                except Exception as e:
                    print(f"Error copying {source_file_path}: {e}")
                    break

# Path to the root folder that contains YOLO1, YOLO2, YOLO3, etc.
source_root = r'D:\araucaria_yolo\datasets'

# Path to the target YOLO_ALL folder
target_root = r'D:\araucaria_yolo\YOLO_ALL'

# Merge the YOLO folders
merge_yolo_folders(source_root, target_root)

#% Exclude non-matching files

import os

def find_non_matching_files(images_folder, labels_folder, image_extension=".jpg", label_extension=".txt"):
    # Get a set of filenames (without extensions) from the 'images' folder
    image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(image_extension))

    # Get a set of filenames (without extensions) from the 'labels' folder
    label_files = set(os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(label_extension))

    # Find files that are in 'images' but not in 'labels'
    non_matching_images = image_files - label_files

    # Find files that are in 'labels' but not in 'images'
    non_matching_labels = label_files - image_files

    return non_matching_images, non_matching_labels

def remove_non_matching_files(images_folder, labels_folder, non_matching_images, non_matching_labels, image_extension=".jpg", label_extension=".txt"):
    # Remove non-matching image files
    for file in non_matching_images:
        image_path = os.path.join(images_folder, file + image_extension)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Removed non-matching image file: {image_path}")

    # Remove non-matching label files
    for file in non_matching_labels:
        label_path = os.path.join(labels_folder, file + label_extension)
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Removed non-matching label file: {label_path}")

def main():
    # Paths to the 'images' and 'labels' folders
    images_folder = r'D:\araucaria_yolo\/YOLO_ALL/images/val'
    labels_folder = r'D:\araucaria_yolo\/YOLO_ALL/labels/val'

    # Find non-matching files
    non_matching_images, non_matching_labels = find_non_matching_files(images_folder, labels_folder)

    print(f"Non-matching images: {non_matching_images}")
    print(f"Non-matching labels: {non_matching_labels}")

    # Remove non-matching files
    remove_non_matching_files(images_folder, labels_folder, non_matching_images, non_matching_labels)

    print("Non-matching files removed.")
    
    # Paths to the 'images' and 'labels' folders
    images_folder = r'D:\araucaria_yolo\YOLO_ALL/images\train'
    labels_folder = r'D:\araucaria_yolo\YOLO_ALL\labels\train'

    # Find non-matching files
    non_matching_images, non_matching_labels = find_non_matching_files(images_folder, labels_folder)

    print(f"Non-matching images: {non_matching_images}")
    print(f"Non-matching labels: {non_matching_labels}")

    # Remove non-matching files
    remove_non_matching_files(images_folder, labels_folder, non_matching_images, non_matching_labels)

    print("Non-matching files removed.")
    
if __name__ == '__main__':
    main()
    
#%% RANDOMLY SHOW IMAGES AND MASKS  
  
import os
import cv2
import random
import numpy as np

# Paths to images and labels
images_folder = r'D:\araucaria_yolo\YOLO_ALL\images\train'
labels_folder = r'D:\araucaria_yolo\YOLO_ALL\labels\train'

# Get list of image files
image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

# Randomly select 5 images
random.seed(42)  # Ensures reproducibility
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

        class_id = int(parts[0])
        points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

        # Convert YOLO format (relative) to absolute pixel coordinates
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(int)

        # Draw the polygon in red
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    
    return image

# Display images in a pop-up window
for img_file in selected_images:
    img_path = os.path.join(images_folder, img_file)
    label_path = os.path.join(labels_folder, img_file.replace('.jpg', '.txt'))

    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a copy and overlay red masks
    masked_image = draw_yolo_mask(image.copy(), label_path, color=(0, 0, 255))

    # Show the image in a pop-up window
    cv2.imshow(f"Original - {img_file}", image)
    cv2.imshow(f"Masked - {img_file}", masked_image)

    # Wait for a key press before moving to the next image
    print(f"Press any key to continue to the next image: {img_file}")
    cv2.waitKey(0)

    # Destroy windows before showing the next pair
    cv2.destroyAllWindows()
    sys.exit()