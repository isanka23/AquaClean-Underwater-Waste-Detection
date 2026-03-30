import os
import torch
import time
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.datasets import CocoDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ==========================================================
# 
# ==========================================================
VAL_IMG_DIR = r"D:\4th year\fyp\under water waste detection\trash_ICRA19\trash_ICRA19\dataset\val"
VAL_ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_val.json" 
WEIGHTS_PATH = r"D:\4th year\fyp\AquaClean_Project\object_detecion\test_weights\detr_best++.pth"

# ==========================================================
# 2. DATA PREPARATION 
# ==========================================================
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

print("📊 Loading Validation Dataset...")
val_dataset = CocoDetection(VAL_IMG_DIR, VAL_ANN_FILE)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ==========================================================
# 3. MODEL LOADING 
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔄 Initializing Model on {device.type.upper()}...")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,
    id2label={0: "plastic", 1: "rov", 2: "bio"},
    label2id={"plastic": 0, "rov": 1, "bio": 2},
    ignore_mismatched_sizes=True
).to(device)

checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Weights loaded successfully: {os.path.basename(WEIGHTS_PATH)}")

# ==========================================================
# 4. EVALUATION LOOP (Optimization for High Accuracy)
# ==========================================================
map_metric = MeanAveragePrecision(iou_type="bbox")
inference_times = []

print(f"🚀 Benchmarking on {len(val_dataset)} images...")

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="🔍 Validating", unit="img"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        target_sizes = batch["original_sizes"].to(device)

        start_time = time.time()
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        inference_times.append(time.time() - start_time)

        val_preds = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.001)
        
        val_targets = [{"boxes": torch.tensor([[obj['bbox'][0], obj['bbox'][1], obj['bbox'][0]+obj['bbox'][2], obj['bbox'][1]+obj['bbox'][3]] for obj in rt], dtype=torch.float32).to(device) if rt else torch.empty((0, 4), device=device),
                        "labels": torch.tensor([obj['category_id'] for obj in rt], dtype=torch.int64).to(device) if rt else torch.empty((0,), dtype=torch.int64, device=device)} for rt in batch['raw_targets']]
        
        map_metric.update(val_preds, val_targets)

results = map_metric.compute()
mAP_50 = results['map_50'].item()
recall = results['mar_100'].item()
precision = results['map'].item() 

avg_inf_time = (sum(inference_times) / len(inference_times)) * 1000
fps = 1000 / avg_inf_time

# ==========================================================
# ==========================================================
print("\n" + "="*60)
print(f"📊 AQUACLEAN DETR FINAL EVALUATION (HIGH ACCURACY)")
print("-" * 60)
print(f"{'mAP @ 0.5 (Accuracy)':<30} | {mAP_50:.4f}")
print(f"{'Recall (R)':<30} | {recall:.4f}")
print(f"{'Precision (P)':<30} | {precision:.4f}")
print(f"{'Avg Inference Time':<30} | {avg_inf_time:.2f} ms")
print(f"{'FPS':<30} | {fps:.2f}")
print("="*60)