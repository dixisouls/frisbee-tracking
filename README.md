# Frisbee Detection using Faster R-CNN

A computer vision project that uses **Faster R-CNN** with **ResNet50 backbone** to detect flying discs (frisbees) in images and videos. Built with **PyTorch** and **OpenCV**, this model can detect and track frisbees in real-time with bounding box visualization and trajectory tracking.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)

---

## Overview

This project aims to detect flying discs in both images and videos using deep learning. The system utilizes a Faster R-CNN model pre-trained on COCO dataset and fine-tuned on custom frisbee data to provide accurate detection with real-time performance.

### Key Workflow:
1. **Custom Dataset** creation with annotated frisbee images
2. **Transfer Learning** using pre-trained Faster R-CNN
3. **Real-time Detection** in videos with trajectory tracking
4. **Performance Optimization** for smooth video processing

---

## Features

- **Faster R-CNN with ResNet50** for robust object detection
- **Custom Dataset Class** for frisbee image processing
- **Real-time Video Processing** with OpenCV
- **Trajectory Tracking** with motion path visualization
- **Flexible Input Support** for both images and videos
- **Performance Monitoring** with inference time tracking

---

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.9.0 or higher
- OpenCV
- Torchvision
- PIL
- NumPy

Clone the repository:
```bash
git clone https://github.com/dixisouls/frisbee-tracking.git
cd frisbee-tracking
```

Install the required libraries:
```bash
pip install torch torchvision opencv-python pillow numpy
```

---

## Usage

### Training

To train the detection model:
```python
python train.py
```

The training script:
- Loads the custom FrisbeeDataset
- Initializes the Faster R-CNN model
- Trains for specified epochs
- Saves the model weights

### Inference

For inference on images:
```python
from inference import detect_image

detect_image(
    image_path='test.jpg',
    output_path='result.jpg',
    thickness=3,
    threshold=0.5
)
```

For video processing:
```python
from inference import detect_video

detect_video(
    video_path='test.mp4',
    output_path='output.avi',
    thickness=3,
    threshold=0.5
)
```

---

## Project Structure
```
frisbee-detection/
├── frisbee_model.ipynb     # Training notebook
├── inference.py            # Inference implementation
├── Frisbee_Data/          
│   ├── images/            # Training images
│   └── annotations/       # XML annotations
├── saved_models/
│   └── frisbee_model.pth  # Trained model weights
└── README.md              # Project documentation
```

---

## Configuration

Model parameters and configurations can be adjusted:

- **Model Architecture**
  - Pre-trained Faster R-CNN with ResNet50 backbone
  - FPN (Feature Pyramid Network)
  - 2 classes (background, frisbee)

- **Training Parameters**
  - Learning rate: 0.001
  - Momentum: 0.9
  - Weight decay: 0.0005
  - Step LR scheduler with step size 3
  - Batch size: 2

- **Inference Parameters**
  - Detection threshold: 0.5
  - Box thickness: 3
  - Video tracking points: 50 (deque maxlen)

---

## Results

The model demonstrates robust detection and tracking capabilities:

- **Real-time Detection**: Achieves efficient inference times suitable for video processing
- **Trajectory Tracking**: Successfully tracks frisbee path during flight
- **Accuracy**: High precision in varying lighting conditions and backgrounds
