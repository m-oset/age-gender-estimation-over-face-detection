# age-gender-estimation-over-face-detection
Group university project.

## Overview

This project implements a complete Deep Learning pipeline for real-time Age and Gender Prediction from faces detected in video streams or images. The system is designed as a two-stage process: first, localizing faces within the frame, and second, performing multi-task classification/regression on the detected regions.

### 1. Face Detection Module (Implemented by other team members)

This initial module is responsible for localizing human faces within the input image.
* Approach: A Sliding Window Convolutional Neural Network (CNN) detector was implemented and trained to classify image patches from multiple scales as either containing a face or background (positive/negative samples).
* Dataset: The detector was trained using the widely recognized WIDERFace dataset.
* Post-Processing: Non-Maximum Suppression (NMS) was used to refine the overlapping output bounding boxes.

### 2. Multi-Task Classification/Regression Module (My Core Contribution)

This final module processes the cropped face regions provided by the detection stage to estimate demographic attributes.
* Architecture: ResNet CNN architecture implemented from scratch in PyTorch, utilizing custom Residual Blocks, and Adam optimizer.
* Multi-Task Heads: The network uses two independent prediction heads for concurrent inference:
  * Gender Classification (Binary): Optimized with Binary Cross-Entropy Loss (nn.BCELoss).
  * Age Regression (Numerical): Optimized with Mean Squared Error Loss (nn.MSELoss).
* Dataset: The classification model was trained and validated on the UTKFace dataset.

### Results

| Task | Metric | Validation Result |
| :--- | :--- | :--- |
| **Gender Classification** | Accuracy | 72% |
| **Age Regression** | Mean Absolute Error (MAE) | 9.14 years |
