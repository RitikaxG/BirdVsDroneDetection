# Bird vs Drone Detection Using YOLOv8

This repository contains a project for detecting birds and drones using the YOLOv8 deep learning model. The model is trained on a custom dataset and fine-tuned for improved accuracy. The project also includes steps for pruning, quantizing, and exporting the trained model to ONNX format for deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Aim of the Project](#aim-of-the-project)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Pruning and Quantization](#pruning-and-quantization)
- [Model Export](#model-export)
- [Usage](#usage)
- [Results](#results)


## Project Overview
The aim of this project is to differentiate between birds and drones using computer vision techniques. We leverage a pre-trained YOLOv8 model, fine-tune it on our dataset, and optimize it through techniques such as pruning and quantization to make it suitable for real-time applications.

## Aim of the Project
1. **Bird vs Drone Detection Using YOLOv8**
   - The primary objective is to **develop a model that can differentiate between birds and drones** using deep learning techniques.
   - This project aims to utilize **YOLOv8**, a powerful object detection model, which is pre-trained and then fine-tuned to identify birds and drones in real-time.

2. **Leveraging Pre-trained Model (YOLOv8)**
   - The project starts by using a **pre-trained YOLOv8 model** (`yolov8m.pt`). Leveraging pre-trained models helps in quickly adapting the model to new tasks, especially when the amount of available data is limited or for quicker training.

3. **Dataset Preparation**
   - The model is trained on a **custom dataset** consisting of images labeled as either "Bird" or "Drone". The dataset is organized into training and validation folders to facilitate model training and performance evaluation.
   - The dataset is hosted by **Roboflow**, and paths are provided to YOLOv8 through a `data.yaml` configuration file.

4. **Training the Model**
   - The pre-trained YOLOv8 model is **fine-tuned** on the custom dataset to adapt it specifically for bird vs drone detection.
   - The model is trained for **20 epochs** with a small learning rate to avoid overfitting and ensure better adaptation to the new dataset.

5. **Pruning for Optimization**
   - **Pruning** is a key step used to **optimize the model** for deployment, reducing the number of parameters and making it suitable for real-time applications.
   - The model is pruned in two steps:
     - **Structured Pruning** removes entire filters or channels to simplify computations and reduce model size.
     - **Unstructured Pruning** removes specific weights, further optimizing the model without compromising accuracy significantly.
   - After pruning, **fine-tuning** is performed to help the model regain any lost accuracy.

6. **Quantization for Deployment**
   - Quantization is performed to convert the model from 32-bit to 8-bit, making it faster and lighter for **deployment in resource-constrained environments** like edge devices or embedded systems.

7. **Model Export to ONNX**
   - The pruned and fine-tuned model is then **exported to ONNX format**, making it easy to deploy across various platforms.
   - ONNX (Open Neural Network Exchange) provides **cross-framework compatibility**, which means the model can be used with different deep learning frameworks and on various devices without needing to re-train.

8. **Results Evaluation**
   - The performance of the model is evaluated using metrics like **Mean Average Precision (mAP)**.
   - The goal is to achieve a model that works in **real-time**, with high accuracy while also being efficient enough for **practical deployment** in real-world scenarios.


## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- OpenCV
- Albumentations (for data augmentation)
- Matplotlib
- YOLOv8 (via `ultralytics` library)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BirdVsDroneDetection
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
The dataset used in this project is hosted by Roboflow. To use the dataset:
1. Download the dataset using the provided Roboflow link.
2. Place the dataset in the `BirdVsDroneDetection.v2i.yolov8/` directory.
3. Make sure the directory structure follows:
   ```
   BirdVsDroneDetection.v2i.yolov8/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── valid/
     │   ├── images/
     │   └── labels/
   ```
4. The dataset paths are specified in `data.yaml`, which is used by YOLO for training.

## Training the Model
To train the YOLOv8 model on the dataset:

1. Load the pre-trained YOLOv8 model:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8m.pt')
   ```

2. Fine-tune the model on the custom dataset:
   ```python
   model.train(data='BirdVsDroneDetection.v2i.yolov8/data.yaml', epochs=20, imgsz=640, batch=8, device='cuda', lr0=0.001)
   ```

- **Epochs**: Set to 20 to fine-tune the model on the new dataset.
- **Learning Rate**: A smaller learning rate (0.001) is used for fine-tuning.

## Pruning and Quantization

### Structured and Unstructured Pruning
Pruning is used to reduce the number of parameters in the model, thus optimizing it for real-time applications.

1. **Structured Pruning**: Prune entire filters or channels to simplify computation.
   ```python
   import torch.nn.utils.prune as prune
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Conv2d):
           prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)
   ```

2. **Unstructured Pruning**: Further prune specific weights.
   ```python
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Conv2d):
           prune.l1_unstructured(module, name="weight", amount=0.3)
   ```

### Fine-tuning After Pruning
It is recommended to fine-tune the pruned model to regain lost accuracy.

## Model Export
After training and pruning, the model is exported to ONNX format for deployment:

```python
import torch

dummy_input = torch.randn(1, 3, 640, 640)
model.export(dummy_input, file_name='student_model.onnx', opset_version=11)
```
- **ONNX Export**: Converts the PyTorch model into ONNX, making it suitable for deployment across different environments.

## Usage
To use the trained model for inference:

1. Set the model to evaluation mode:
   ```python
   model.eval()
   ```
2. Preprocess the image using OpenCV:
   ```python
   image = preprocess_image('path/to/image.jpg')
   result = model(image)
   ```
3. Interpret the results and visualize them.

## Results
- The model achieved significant improvement in real-time detection by applying pruning and quantization.
- mAP (Mean Average Precision) values are calculated to evaluate the performance.


