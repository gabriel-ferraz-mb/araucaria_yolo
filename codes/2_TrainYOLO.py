import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import geopandas as gpd  # Adicionado
import torch
from ultralytics import YOLO


if __name__ == "__main__":
    # Ensure that multiprocessing uses 'spawn' method on Windows
    torch.multiprocessing.set_start_method("spawn", force=True)


    # Path to your dataset
    Dataset = r'C:\araucaria_yolo\datasets\YOLO_CHM_Composite'

    # Path to your labels shapefile (assuming it's the same as used for data generation)
    labels_path = r"C:\araucaria_yolo\mascaras\mascaras_merge_bbox.shp"
    mask_gdf = gpd.read_file(labels_path)
    species = sorted(mask_gdf["tree_name"].unique().tolist())

    # Generate the names section for the YAML file dynamically
    names_yaml = "\n".join([f"      {i}: {specie}" for i, specie in enumerate(species)])

    # Define the data YAML content
    yaml_data = f"""
    path:  '{os.path.abspath(Dataset)}'
    train: 'images/train'
    val:   'images/val'

    names:
{names_yaml}
    """


    # Save the YAML configuration to a file
    save_dir = os.path.join(Dataset, 'model_lif.yaml')


    with open(save_dir, "w") as f:
        f.write(yaml_data)


    # -------------------------------------------------------------------------
    #  Specify the path to your PRETRAINED YOLO model weights below.
    #  For segmentation: 'yolov8x-seg.pt', 'yolov8n-seg.pt', etc.
    #  For detection only: 'yolov8x.pt', 'yolov8n.pt', etc.
    # -------------------------------------------------------------------------
    pretrained_weights = 'yolov8x-seg.pt'

    # Load the YOLO model with pretrained weights
    # model = YOLO(r"C:\araucaria_yolo\runs\segment\experiment_7\weights\best.pt")
    model = YOLO(r"C:\araucaria_yolo\runs\segment\experiment_7\weights\best.pt") 
    device = 0 #if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo para treinamento: {device}")

    # Start training with the parameters defined
        # Start training with the parameters defined
        # Com a RTX 5080, você pode usar batch=32 ou até 64 tranquilamente

    results = model.train(
        data=save_dir,   # YAML file specifying dataset paths + class names
        epochs=250,
        patience=40,
        mosaic=1,
        imgsz=640,
        resume = False, 
        name='experiment_8',           # Custom experiment name
        plots=True,
        batch=16,
        workers=8,         
        amp=True,           
        exist_ok=True,
        save_period=-1,
        # -----------------------------
        # Augmentation Hyperparameters
        # -----------------------------
        device=device,
        # scale=0.8,          # Ajustado
        # shear=5,
        perspective=0.0,
        flipud=0.5,         # Ajustado
        fliplr=0.5,
        # hsv_h=0.010,        # Ajustado
        # hsv_s=0.5,          # Ajustado
        # hsv_v=0.3,          # Ajustado
        # erasing=0.2,  
        degrees=10,         # Random rotation (+/- degrees)
        translate=0.1,      # Translation fraction (0.1 = 10%)
        )
