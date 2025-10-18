# Image Processing Coursework – Data Science

## Overview

This project, part of the second year **Data Science** module at Durham University (2024–25 academic year), focuses on enhancing the quality of corrupted driving images through classical **image processing techniques**. The goal is to improve both **visual quality** and **classifier performance** in a self-driving car scenario.

The provided dataset contains 100 images affected by noise, warping, brightness imbalance, and missing regions. The project applies a full enhancement pipeline (including rotation correction, perspective warping, denoising, gamma correction, colour balancing, and inpainting) to restore image quality before classification.

---

## Key Features

* **Rotation & Warping:** Aligns and corrects distorted images using adaptive thresholding, edge detection, and perspective transforms.
* **Denoising:** Applies section-based **Non-Local Means filtering** for strong noise removal with minimal blurring.
* **Brightness & Contrast:** Uses **adaptive gamma correction** for consistent illumination across images.
* **Colour Correction:** Employs the **Max-RGB (White Patch)** method to balance colour channels.
* **Inpainting:** Detects and fills missing regions using **Hough Circle Transform** or connected component analysis with OpenCV’s TELEA method.

---

## Usage

1. Place your input images in a directory (e.g. `./driving_images/`).
2. Run the main script:

   ```bash
   python main.py ./driving_images/
   ```
3. The processed images will be saved in a new folder called **Results/** (automatically created).
4. Evaluate your results using the provided `classify.py` and `classifier.model` files.

---

## Repository Structure

```
├── main.py              # Image processing pipeline implementation
├── classify.py          # Code for image classifying
├── driving_images/      # Input images (not included in submission)
├── report.pdf           # Coursework report detailing methods and results
└── README.md            # Project overview
```

---

## Techniques Summary

| Stage                     | Method                          | Classifier Accuracy |
| ------------------------- | ------------------------------- | ------------------- |
| Raw images                | —                               | 55%                 |
| Rotation + Warping        | Affine & Perspective transforms | 91%                 |
| Denoising                 | Non-Local Means (sectioned)     | 90–96%              |
| Gamma + Colour Correction | Adaptive gamma, Max-RGB         | 93%                 |
| Inpainting                | Hough Circle + TELEA            | 93%                 |


