 DRIVE Dataset Retina Vessel Segmentation

This repository contains code for retina vessel segmentation using the DRIVE dataset. The project applies various image processing techniques, data augmentation, feature extraction, and deep learning-based UNet models for vessel segmentation from retinal images. The repository also includes methods for model evaluation, performance metrics, and visualization.

## Features

- **Image Preprocessing**: Resizes and normalizes images.
- **Data Augmentation**: Includes rotation, flipping, and complex transformations.
- **Patch Extraction**: Extracts patches from images for better training.
- **Feature Extraction**: Multiple methods like green channel extraction, CLAHE, grayscale, LBP, HSV, and more.
- **UNet Models**: Standard UNet and Dense UNet for segmentation.
- **Post-processing**: Includes binary closing and opening to improve segmentation results.
- **Evaluation**: Confusion matrix, ROC curves, AUC, and precision-recall scores.

## Requirements

To run this code, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib scikit-image scikit-learn tensorflow imageio opencv-python
