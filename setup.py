#!/usr/bin/env python3
import os
import torch
import subprocess
from pathlib import Path

def download_yolo_model():
    """Download YOLOv5 model and save locally"""
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'yolov5s.pt'
    
    if not model_path.exists():
        print("Downloading YOLOv5s model...")
        # Clone YOLOv5 repository
        if not os.path.exists('yolov5'):
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
        
        # Download the model
        url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
        torch.hub.download_url_to_file(url, str(model_path))
        print(f"Model downloaded to {model_path}")
    else:
        print("Model already exists locally")

if __name__ == "__main__":
    try:
        download_yolo_model()
        print("Setup completed successfully!")
    except Exception as e:
        print(f"Error during setup: {e}")