from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from dataset import creat_yolo_dataset 
from pathlib import Path
import os

BASE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/06_yolo")

xml_dir = BASE_DIR / "data/VOCdevkit/VOC2007/Annotations"
image_dir = BASE_DIR / "data/VOCdevkit/VOC2007/JPEGImages"

train_img_dir = BASE_DIR / "data/YOLO/images/train"
val_img_dir = BASE_DIR / "data/YOLO/images/val"
train_label_dir = BASE_DIR / "data/YOLO/labels/train"
val_label_dir = BASE_DIR / "data/YOLO/labels/val"

class_names = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

file_list = [
    filename for filename in os.listdir(xml_dir)
    if filename.endswith(".xml")
]

train_files, val_files = train_test_split(
    file_list,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

creat_yolo_dataset(
    train_files, 
    image_dir, 
    train_img_dir, 
    xml_dir, 
    train_label_dir, 
    class_names
)

creat_yolo_dataset(
    val_files, 
    image_dir, 
    val_img_dir, 
    xml_dir, 
    val_label_dir, 
    class_names
)
