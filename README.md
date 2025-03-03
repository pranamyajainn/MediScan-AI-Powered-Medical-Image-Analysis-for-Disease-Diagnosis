# Medical Image Feature Extraction and Classification

This script performs medical image processing to extract texture, shape, and color features, and then uses a Random Forest Classifier to train a model.

## Overview

The script loads a medical image, preprocesses it (resize, normalize, blur, segment), extracts features using GLCM, contour analysis, and color histograms, and then trains a Random Forest model on these features. It also visualizes key processing steps and feature distributions.

## Dependencies

-   `cv2` (OpenCV): For image processing.
-   `numpy`: For numerical operations.
-   `matplotlib.pyplot`: For image and histogram visualization.
-   `skimage.feature`: For GLCM feature extraction.
-   `sklearn.ensemble`: For Random Forest Classifier.
-   `sklearn.model_selection`: For splitting the dataset.
-   `sklearn.metrics`: For evaluating the model.
-   `joblib`: For saving the trained model.

Install the required packages using pip:

```bash
pip install opencv-python numpy matplotlib scikit-image scikit-learn joblib
