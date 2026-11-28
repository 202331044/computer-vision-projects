from pathlib import Path

class Config:

    # =====================================
    # PASCAL VOC 2007
    # =====================================
    XML_2007_DIR = Path("/content/dataset/VOCdevkit/VOC2007/Annotations")
    IMG_2007_DIR = Path("/content/dataset/VOCdevkit/VOC2007/JPEGImages")

    # =====================================
    # PASCAL VOC 2012
    # =====================================
    XML_2012_DIR = Path("/content/dataset/VOCdevkit/VOC2012/Annotations")
    IMG_2012_DIR = Path("/content/dataset/VOCdevkit/VOC2012/JPEGImages")

    # =====================================
    # PASCAL VOC 2007 TEST
    # =====================================
    XML_2007_TEST_DIR = Path("/content/dataset/test/VOCdevkit/VOC2007/Annotations")
    IMG_2007_TEST_DIR = Path("/content/dataset/test/VOCdevkit/VOC2007/JPEGImages")

    # =====================================
    # YOLO DATASET BASE DIRECTORY
    # =====================================
    BASE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/06_yolo")

    YOLO_DIR = BASE_DIR / "data/YOLO"

    DATA_YAML = YOLO_DIR / "data.yaml"

    # =====================================
    # TRAIN / VALIDATION / TEST IMAGES
    # =====================================
    TRAIN_IMG_DIR = YOLO_DIR / "images/train"
    VAL_IMG_DIR = YOLO_DIR / "images/val"
    TEST_IMG_DIR = YOLO_DIR / "images/test"

    # =====================================
    # TRAIN / VALIDATION / TEST LABELS
    # =====================================
    TRAIN_LABEL_DIR = YOLO_DIR / "labels/train"
    VAL_LABEL_DIR = YOLO_DIR / "labels/val"
    TEST_LABEL_DIR = YOLO_DIR / "labels/test"

    # =====================================
    # PASCAL VOC CLASSES
    # =====================================
    CLASS_NAMES = [
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

    NUM_CLASSES = len(CLASS_NAMES)