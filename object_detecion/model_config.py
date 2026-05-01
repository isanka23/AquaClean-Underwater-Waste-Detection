import os
import csv
import torch
torch.set_float32_matmul_precision('high')
import time
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.datasets import CocoDetection
from comet_ml import Experiment
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# --- 0. COMET ML SETUP ---
experiment = Experiment(
    api_key="P1J4eRFU4Hx90OAG8Da7Ci9GW",
    project_name="FYP-AquaClean",
)
experiment.set_name("DETR_Final_Boosted_Run")

# --- 1. CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 20

TRAIN_IMG_DIR = "/content/dataset_root"
TRAIN_ANN_FILE = "/content/drive/MyDrive/dataset_combine_train/final_train_with_rov_boost.json"
VAL_IMG_DIR = "/content/dataset_root"
VAL_ANN_FILE = "/content/drive/MyDrive/dataset_combine_train/seaclear_test.json"

OUTPUT_DIR = "/content/drive/MyDrive/dataset_combine_train/AquaClean_Models_Final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_log_path = os.path.join(OUTPUT_DIR, "training_metrics_log.csv")
latest_checkpoint_path = os.path.join(OUTPUT_DIR, "detr_latest.pth")
best_checkpoint_path = os.path.join(OUTPUT_DIR, "detr_best.pth")

# --- 2. DATA PREPARATION ---
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def collate_fn(batch):
    images = [item[0] for item in batch]
    formatted_annotations = [{"image_id": item[1][0]['image_id'] if item[1] else i, "annotations": item[1]} for i, item in enumerate(batch)]
    encoding = processor(images=images, annotations=formatted_annotations, return_tensors="pt")
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': encoding['labels'],
        'original_sizes': torch.tensor([(img.height, img.width) for img in images]),
        'raw_targets': [item[1] for item in batch]
    }

print("📊 Loading datasets...")
train_dataset = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANN_FILE)
val_dataset = CocoDetection(VAL_IMG_DIR, VAL_ANN_FILE)

def fix_filenames(dataset):
    for img_id in dataset.coco.imgs:
        original_name = dataset.coco.imgs[img_id]['file_name']
        dataset.coco.imgs[img_id]['file_name'] = os.path.basename(original_name)

fix_filenames(train_dataset)
fix_filenames(val_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=12, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=12, pin_memory=True)

# --- 3. MODEL INITIALIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,
    id2label={0: "plastic", 1: "rov", 2: "bio"},
    label2id={"plastic": 0, "rov": 1, "bio": 2},
    ignore_mismatched_sizes=True
).to(device)

# --- BACKBONE FINE-TUNING (LAYER 4 UNFREEZE) ---
for name, param in model.named_parameters():
    if "backbone.conv_encoder.model.layer4" in name:
        param.requires_grad = True # Underwater distortion 
    elif "backbone" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

print("-" * 50)
print("✅ CONFIGURATION SUCCESSFUL!")
print(f"📂 Training on {len(train_dataset)} images | Validating on {len(val_dataset)} images")
print(f"💻 Device: {device.type.upper()} | Classes: {model.config.id2label}")
print("-" * 50)
