import torch

# --- PROJECT PATHS ---
# Update this to where your model.py is located
MODEL_DIR = r"D:\4th year\fyp\AquaClean_Project\enhancment"
GAN_WEIGHTS_PATH = r"D:\4th year\fyp\AquaClean_Project\enhancment\results_gan\gan_best_checkpoint.pth"

# Currently active model weights
DETR_WEIGHTS_PATH = r"D:\4th year\fyp\AquaClean_Project\object_detecion\test_weights\detr_best++.pth"
# --- MODEL SETTINGS ---
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.70

# --- CLASS LABELS ---
id2label = {0: 'plastic', 1: 'bio', 2: 'rov'} 
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# --- UI VISUALS ---
BOX_COLORS = {
    'plastic': (255, 0, 0),    # Red
    'bio': (0, 255, 0),        # Green
    'rov': (255, 255, 0)       # Yellow
}
TEXT_COLOR = (255, 255, 255)   # White