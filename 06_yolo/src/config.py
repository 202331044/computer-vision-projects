import torch

class Config:

    # =========================
    # Model Configuration
    # =========================
    NUM_BOXES = 2
    NUM_CLASSES = 20
    GRID_SIZE = 7

    TARGET_WIDTH = 448
    TARGET_HEIGHT = 448

    # =========================
    # Training Configuration
    # =========================
    EPOCHS = 1
    BATCH_SIZE = 32

    # =========================
    # Optimizer Configuration
    # =========================
    OPTIMIZER = "SGD"

    INITIAL_LR = 1e-3
    MAX_LR = 1e-2
    MIN_LR = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # =========================
    # LR Scheduler
    # =========================
    WARMUP_EPOCHS = 10

    LR_STAGE_1 = 75
    LR_STAGE_2 = 105
    LR_STAGE_3 = 135

    # =========================
    # Regularization
    # =========================
    DROP_OUT = 0.5

    # =========================
    # Evaluation
    # =========================
    # Minimum detection score required to retain a predicted bounding box.
    # Calculated as confidence multiplied by class probability.
    SCORE_THRESHOLD = 0.4

    # IoU threshold used by Non-Maximum Suppression (NMS).
    # A detection is suppressed if its IoU with a higher-scoring
    # detection of the same class exceeds this threshold.
    NMS_IOU_THRESHOLD = 0.5
    
    # IoU threshold used to determine correct detections during mAP evaluation.
    MAP_IOU_THRESHOLD = 0.5
    
    # =========================
    # Device
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Pretrained Weights
    # =========================
    WEIGHT_PATH = "../weights/extraction.weights"
    NUM_PRETRAINED_CONV_LAYERS = 20

    # =========================
    # Checkpoint
    # =========================
    # Path for saving and resuming the latest training state.
    CHECKPOINT_PATH = 'checkpoint.pth'
    # Path for saving the model with the lowest validation loss.
    BEST_MODEL_PATH = 'best_model.pth'
    