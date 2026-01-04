from ultralytics import YOLO
from models.hierarchical_loss import hierarchical_loss
import torch

def main():
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=2   # important for Windows
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()