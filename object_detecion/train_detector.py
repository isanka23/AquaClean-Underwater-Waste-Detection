import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.datasets import CocoDetection

# --- 1. CONFIGURATION ---
LIMIT_IMAGES = 400      # Subset size for CPU training
BATCH_SIZE = 2          
EPOCHS = 15           
LEARNING_RATE = 1e-4

# Paths for your specific setup
IMG_DIR = r"D:\4th year\fyp\trash_ICRA19\trash_ICRA19\dataset\train"
ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_train.json"
OUTPUT_DIR = r"D:\4th year\fyp\AquaClean_Project\object_detecion"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. DATA PREPARATION ---
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encoding = processor(images=images, return_tensors="pt")
    return {
        'pixel_values': encoding['pixel_values'], 
        'pixel_mask': encoding['pixel_mask'], 
        'labels': labels
    }

print("Loading dataset...")
full_dataset = CocoDetection(IMG_DIR, ANN_FILE)

# Convert indices to a list of integers
indices = torch.randperm(len(full_dataset))[:LIMIT_IMAGES].tolist()
train_subset = Subset(full_dataset, indices)

dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print(f"Dataset loaded. Training on {len(train_subset)} images in {len(dataloader)} batches.")

# --- 3. MODEL INITIALIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,  # plastic, rov, bio
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. TRAINING LOOP ---
print("\nStarting Training...")
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    total_batches = len(dataloader)
    
    for idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        
        labels = []
        for img_labels in batch["labels"]:
            if len(img_labels) == 0:
                labels.append({"class_labels": torch.empty(0, dtype=torch.long, device=device), 
                               "boxes": torch.empty((0, 4), dtype=torch.float, device=device)})
                continue
                
            class_labels = torch.tensor([obj["category_id"] for obj in img_labels], dtype=torch.long, device=device)
            
            # --- THE CRITICAL FIX: DYNAMIC NORMALIZATION ---
            raw_boxes = []
            for obj in img_labels:
                x, y, w, h = obj["bbox"]
                
                # Using the true dimensions of Trash-ICRA19
                img_w = 480.0
                img_h = 360.0 
                
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                raw_boxes.append([cx, cy, nw, nh])
            
            boxes = torch.tensor(raw_boxes, dtype=torch.float, device=device)
            labels.append({"class_labels": class_labels, "boxes": boxes})

        # Forward Pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Live Progress Tracking
        if (idx + 1) % 5 == 0 or (idx + 1) == total_batches:
            percent = ((idx + 1) / total_batches) * 100
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{idx+1}/{total_batches}] ({percent:.1f}%) | Loss: {loss.item():.4f}", end="\r")
            
    avg_epoch_loss = epoch_loss / total_batches
    print(f"\n-> Epoch {epoch+1} Completed. Average Loss: {avg_epoch_loss:.4f}")
    
    # Checkpoint saving after every epoch
    save_path = os.path.join(OUTPUT_DIR, f"detr_trash_model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)

print(f"\nTraining Complete! Final weights saved to {OUTPUT_DIR}")