YOLO v5
0. Model Execution Environment (HW/SW)
To ensure proper execution of the YOLOv5 model, verify the following hardware and software requirements:

Python Version: The project uses Python 3.12.6. Ensure compatibility by installing the correct version. Python 3.12.6 Download

CUDA Version: The project is set up with CUDA 12.4 for GPU acceleration. CUDA 12.4 Installation Guide

NVIDIA Driver Version: Make sure to install NVIDIA driver version 560.94 to match CUDA 12.4. Use nvidia-smi to verify your driver installation.

YOLOv5 Model Setup
Model Setup: Import necessary libraries, load the YOLOv5 model, and ensure it is running on CUDA (GPU) for faster training.

Multiprocessing and CUDA Memory Management: Ensure multiprocessing support and efficient GPU memory management during training.

Loading and Training the YOLOv5 Model: Load the YOLOv5 model and begin training with specified parameters such as dataset path, number of epochs, batch size, image size, and GPU usage.

YOLOv5
0. Model Execution Environment (HW/SW)
To run YOLOv5, ensure the following environment setup:

Python Version: Python 3.12.6
Torch Version : 2.4.1
CUDA Version: 12.4
CUDNN Version : 9.4
NVIDIA Driver Version: 560.94
YOLOv5 Model File: yolov5m.pt
YOLOv5 Model Setup
1. Model Setup: Import the necessary libraries and move the YOLOv5 model to CUDA (GPU).
2. Multiprocessing and CUDA Memory Management: Enable multiprocessing support and optimize CUDA memory management.
3. Loading and Training the YOLOv5 Model: Load the YOLOv5 model and start training with the specified dataset path, number of epochs, batch size, etc.



# 1. Model Setup
import os
import torch
from yolov5 import YOLO  # Import YOLOv5 library

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

```cmd
python train.py --img 640 --batch 16 --epochs 5000 --patience 50 --data data/data.yaml --cfg models/yolov5s.yaml --weights weights/yolov5m.pt --device 0

```

This command specifies the following parameters:
--img 640: The image size (640x640 pixels).
--batch 16: The batch size (16 images per batch).
--epochs 5000: The total number of training epochs (5000).
--patience 50: The early stopping patience (50 epochs).
--data data/data.yaml: The path to the dataset configuration file.
--cfg models/yolov5s.yaml: The model configuration file for YOLOv5s.
--weights weights/yolov5m.pt: The pre-trained weights file (YOLOv5m).
--device 0: Specifies to use GPU device 0 for training.



YOLOv10
0. Model Execution Environment (HW/SW)
To run YOLOv10, ensure the following environment setup:

Python Version: Python 3.12.6
Torch Version : 2.4.1
CUDA Version: 12.4
CUDNN Version : 9.4
NVIDIA Driver Version: 560.94
YOLOv10 Model File: yolov10m.pt
YOLOv10 Model Setup
1. Model Setup: Load the YOLOv10 model and move it to CUDA (GPU).

2. Multiprocessing and CUDA Memory Management: Enable multiprocessing support and manage CUDA memory.

3. Loading and Training the YOLOv10 Model: Load the YOLOv10 model and start training with the specified settings, including the dataset path, number of epochs, batch size, and more.

```python
# 1. Model Setup
import os
import torch
from ultralytics import YOLO  # Import YOLOv10 library

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Multiprocessing and CUDA Memory Management
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Enable multiprocessing support

    torch.cuda.empty_cache()  # Clear CUDA memory cache

# 3. Loading and Training the YOLOv10 Model
model = YOLO('yolov10m.pt')  # Load YOLOv10 model
model.to('cuda')  # Move model to GPU (CUDA)

# Training the model with specified parameters
model.train(
    data="C:/Users/yourpath/yolov10/ultralytics/data/data.yaml",  # Path to dataset configuration
    epochs=5000,  # Number of training epochs
    patience=50,  # Early stopping patience
    batch=16,  # Batch size for training
    imgsz=640,  # Image size
    device=0  # GPU device ID
)
```