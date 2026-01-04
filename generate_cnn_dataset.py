import os
import cv2
from ultralytics import YOLO

# ===================== CONFIGURATION =====================
YOLO_MODEL = "runs/segment/train5/weights/best.pt"
IMAGE_DIR = "data/images/train"  # Original images
OUT_DIR = "data/cnn"
CONF_THRESHOLD = 0.4

# Hierarchical mapping (example)
HIERARCHY = {
    "Zostera_marina": {
        "binary": "biotic",
        "class": "plant",
        "genus": "zostera",
        "species": "Zostera_marina"
    },
    "Fucus_vesiculosus": {
        "binary": "biotic",
        "class": "algae",
        "genus": "fucus",
        "species": "Fucus_vesiculosus"
    },
    "Pipe": {
        "binary": "abiotic",
        "class": "manmade",
        "genus": "pipe",
        "species": "Pipe"
    }
}

def save_crop(img, box, paths, idx):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return

    for level, path in paths.items():
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(
            os.path.join(path, f"{idx}.jpg"),
            crop
        )

def generate_dataset():
    model = YOLO(YOLO_MODEL)

    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]

    idx = 0
    for img_name in img_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)

        results = model(img, conf=CONF_THRESHOLD)[0]

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            class_name = model.names[int(cls)]
            if class_name not in HIERARCHY:
                continue

            hierarchy = HIERARCHY[class_name]

            # Define paths for train and val
            train_paths = {
                level: os.path.join(OUT_DIR, "train", level, value)
                for level, value in hierarchy.items()
            }
            val_paths = {
                level: os.path.join(OUT_DIR, "val", level, value)
                for level, value in hierarchy.items()
            }

            # Alternate between train and val
            if idx % 5 == 0:
                paths = val_paths
            else:
                paths = train_paths

            save_crop(img, box, paths, idx)
            idx += 1

    print(f"✅ CNN dataset generated: {idx} object crops")

if __name__ == "__main__":
    generate_dataset()
