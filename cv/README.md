# Sentinel-2 Cross-Season Image Matching

Computer vision solution for matching satellite images across different seasons using SIFT feature detection and RANSAC filtering.

## Overview

This project implements a classical computer vision approach to match Sentinel-2 satellite images taken in different seasons. The algorithm uses SIFT (Scale-Invariant Feature Transform) for keypoint detection and RANSAC for robust geometric verification.

## Dataset

The dataset consists of Sentinel-2 satellite images from the same geographical location (tile T36UYA) captured in different seasons:

- Winter (February 2016)
- Summer (June 2016)
- Autumn (September 2019)
- Spring (April 2019)

**Dataset link:** [Google Drive](https://drive.google.com/drive/folders/1AX3zGVndVZUIRNR9A3lFlqRIGrC37vxG?usp=drive_link)

## Project Structure

```
cv/
 data/
    raw/              # Original Sentinel-2 .SAFE folders
    pairs/            # Preprocessed image pairs
    metadata.json     # Dataset metadata
 weights/              # Model weights (N/A for SIFT)
 algorithm.py          # SIFT matching implementation
 inference.py          # Inference script
 demo.ipynb            # Demo notebook
 dataset_preparation.ipynb  # Dataset creation notebook
 requirements.txt      # Python dependencies
 README.md            # This file
```

## Setup

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Installation

Install data from the linked dataset in ./cv/data and requirements from requirements.txt

```bash

pip install -r requirements.txt
```

## Usage

### Dataset Preparation

See `01_dataset_preparation.ipynb` for the complete dataset creation process.

The notebook covers:

- Loading Sentinel-2 images
- Preprocessing
- Creating image pairs from different seasons

### Running Inference

Process a single pair:

```bash
python inference.py --pair_dir ./data/pairs/pair_001 --output_dir ./results
```

Process all pairs:

```bash
python inference.py --pairs_dir ./data/pairs --output_dir ./results
```

### Demo

Run the Jupyter notebook for interactive visualization:

```bash
jupyter notebook demo.ipynb
```

The demo notebook shows:

- Loading and visualizing image pairs
- Running SIFT matching
- Visualizing keypoint matches
- Statistics and analysis

## Algorithm Details

### SIFT Matcher

**Parameters:**

- `ratio_threshold`: 0.8 (Lowe's ratio test)
- `ransac_threshold`: 10.0 (RANSAC reprojection threshold in pixels)
- `min_matches`: 4 (minimum matches required)
- `nfeatures`: 10000 (maximum SIFT features to detect)

**Pipeline:**

1. **Keypoint Detection**: Detect SIFT keypoints in both images
2. **Feature Matching**: Match descriptors using BFMatcher with k=2
3. **Ratio Test**: Filter matches using Lowe's ratio test
4. **RANSAC**: Remove outliers using homography estimation

### Preprocessing

Images are preprocessed with:

- NODATA region filtering (minimum 30% valid pixels)
- Cropping to common valid region between pairs
- Histogram equalization (CLAHE)
- Resizing to 512x512 pixels

### Observations

- Matching between drastically different seasons (winter vs summer/autumn) is challenging due to snow coverage and vegetation changes
- Best results achieved with similar seasons
- RANSAC effectively filters geometric outliers

## Model Weights

This project uses SIFT, a classical computer vision algorithm that does not require training. No model weights are needed.
