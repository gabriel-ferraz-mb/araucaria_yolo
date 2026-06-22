# araucaria_yolo

Detection of *Araucaria angustifolia* individuals in high-resolution aerial imagery using YOLO-based object detection models.

---

## Overview

This repository contains the initial experiments for individual-level detection of *Araucaria angustifolia* — the Paraná pine, a critically endangered species endemic to the Atlantic Forest — using convolutional object detectors applied to aerial orthophotos. The study area covers regions classified as Mixed Ombrophilous Forest (*Floresta Ombrófila Mista*) phytophysiognomy within the state of São Paulo, Brazil.

The work explores multiple input configurations to assess the contribution of spectral and structural information beyond standard RGB imagery, including near-infrared bands (GBNIR) and a Canopy Height Model (CHM) derived from LiDAR data.

This repository represents an early-stage baseline and was followed by more advanced experiments using transformer-based detectors (CanopyRS/DINO) and semantic segmentation approaches (TreeCountSegHeight).

---

## Study Area and Data

- **Location:** Regions classified as Mixed Ombrophilous Forest (*Floresta Ombrófila Mista*) phytophysiognomy, state of São Paulo, Brazil
- **Imagery:** High-resolution RGB and GBNIR aerial orthophotos at 0.15 m/pixel
- **Structural data:** LiDAR-derived Canopy Height Model (CHM) at ~40 cm point sampling
- **Annotations:** Bounding box annotations for *Araucaria angustifolia* crowns, stored as shapefiles
- **Tiles:** Generated from large raster tiles using a sliding-window approach, with train/validation spatial splits

---

## Input Configurations Tested

| Configuration | Channels | Description |
|---------------|----------|-------------|
| RGB | 3 | Standard visible bands (Red, Green, Blue) |
| GBNIR | 3 | Green, Blue, Near-Infrared |
| GB + CHM | 3 | Green, Blue + LiDAR-derived canopy height |

The multi-channel experiments were motivated by the distinctive spectral signature of *Araucaria angustifolia* crowns (umbrella-shaped, dark green) and by the hypothesis that CHM-derived height information would help separate araucaria individuals from the surrounding canopy.

---

## Repository Structure

```
araucaria_yolo/
└── codes/           # Python scripts for preprocessing, training, and evaluation
```

---

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`yolov8` or compatible)
- `rasterio`, `geopandas`, `numpy`, `opencv-python`
- CUDA-compatible GPU (experiments run on RTX 5080, WSL2 / Ubuntu)

Install dependencies:

```bash
pip install ultralytics rasterio geopandas numpy opencv-python
```

---

## Usage

### 1. Prepare tiles

Convert the large raster tiles and shapefile annotations into YOLO-format image/label pairs:

```bash
python codes/prepare_dataset.py \
    --images /path/to/orthophotos/ \
    --annotations /path/to/annotations.shp \
    --output /path/to/dataset/ \
    --mode rgb   # or gbnir, gb_chm
```

### 2. Train

```bash
python codes/train.py \
    --data /path/to/dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --imgsz 640
```

### 3. Evaluate and run inference

```bash
python codes/predict.py \
    --weights /path/to/best.pt \
    --source /path/to/tiles/ \
    --conf 0.3
```

> **Note:** Script names above are illustrative. Check the `codes/` directory for the actual file names and their argument interfaces.

---

## Results (Baseline)

These experiments served as a proof-of-concept and baseline prior to adopting transformer-based detectors. Performance was measured in terms of COCO AP metrics on the held-out validation tiles.

Key findings:
- GBNIR and GB+CHM inputs provided complementary information relative to RGB alone, with CHM being particularly informative for separating araucaria crowns from other tall-canopy species.
- YOLO-based detection achieved reasonable localization but struggled with crown delineation precision in dense canopy conditions.
- These results motivated the transition to DINO-based detection (CanopyRS) and attention-based segmentation (TreeCountSegHeight) in subsequent experiments.

---

## Related Repositories

- **[CanopyRS](https://github.com/lu-liang-geo/CanopyRS)** — DINO Swin-Large detector fine-tuned for araucaria crown detection (advanced experiments)
- **[TreeCountSegHeight](https://github.com/ameliajimenez/TreeCountSegHeight)** — Attention U-Net for tree crown segmentation and height estimation, targeting the full RGB+GBNIR+CHM input configuration

---

## Citation

If you use this code or data in your work, please cite the corresponding publication (in preparation).

---

## License

This project is part of ongoing academic research. Contact the author for usage permissions.

---

## Author

**Gabriel Ferraz**
Remote sensing and forest ecology researcher, Brazil.
GitHub: [@gabriel-ferraz-mb](https://github.com/gabriel-ferraz-mb)
