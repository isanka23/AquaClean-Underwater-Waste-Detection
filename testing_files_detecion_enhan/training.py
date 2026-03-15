from comet_ml import start, login
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from pycocotools.coco import COCO
import os
from PIL import Image
from tqdm import tqdm
from models import JointModel
import argparse

# 1. SETUP & ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--enhancement_weight', type=float, default=0.1)
args = parser.parse_args()

login(api_key="P1J4eRFU4Hx90OAG8Da7Ci9GW") 
experiment = start(project_name="aquaclean-underwater-waste")
experiment.log_parameters(vars(args))

# 2. LOSS FUNCTIONS

# Color Constancy Loss
def L_color(x):   
    mean_rgb = torch.mean(x, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    Drg = torch.pow(mr - mg, 2)
    Drb = torch.pow(mr - mb, 2)
    Dgb = torch.pow(mb - mg, 2)
    return torch.mean(torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5))

# Exposure Control Loss
def L_exp(x, mean_val=0.6):
    x = torch.mean(x, 1, keepdim=True)
    avg = F.avg_pool2d(x, 16)
    return torch.mean(torch.pow(avg - mean_val, 2))

# Spatial Consistency Loss
def L_spa(org, enh):
    kernel = torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).expand(1, 1, 3, 3).to(org.device)
    org_mean = torch.mean(org, 1, keepdim=True)
    enh_mean = torch.mean(enh, 1, keepdim=True)
    org_grad = F.conv2d(org_mean, kernel, padding=1)
    enh_grad = F.conv2d(enh_mean, kernel, padding=1)
    return torch.mean(torch.pow(org_grad - enh_grad, 2))

# 3. DATA LOADING
class UnderwaterDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, processor):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        # only 100 images
        self.ids = self.ids[:100]
        self.img_folder = img_folder
        self.processor = processor

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        
        formatted_annotations = {'image_id': img_id, 'annotations': target}
        encoding = self.processor(images=image, annotations=formatted_annotations, return_tensors="pt")
        
        return encoding["pixel_values"].squeeze(0), encoding["labels"][0]

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return {"pixel_values": torch.stack([item[0] for item in batch])}, [item[1] for item in batch]

# 4. MAIN TRAINING & VALIDATION LOOP
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_img_dir = r"D:\4th year\fyp\trash_ICRA19\trash_ICRA19\dataset\train"
    train_ann_file = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_train.json"
    
    val_img_dir = r"D:\4th year\fyp\trash_ICRA19\trash_ICRA19\dataset\val"
    val_ann_file = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_val.json" 

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"height": 800, "width": 800}) #Resizing & NORMALIZE: Scale 0-255 pixels to 0-1
    model = JointModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Separate Loaders for Train and Val
    train_dataset = UnderwaterDataset(train_img_dir, train_ann_file, processor)
    val_dataset = UnderwaterDataset(val_img_dir, val_ann_file, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (encoding, labels) in enumerate(train_loop):
            pixel_values = encoding["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            outputs, enh_imgs, A = model(pixel_values=pixel_values, labels=labels)
            
            det_loss = outputs.loss
            en_loss = L_color(enh_imgs) + L_exp(enh_imgs) + L_spa(pixel_values, enh_imgs)
            total_loss = det_loss + (args.enhancement_weight * en_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log to Comet
            experiment.log_metric("train_total_loss", total_loss.item())
            train_loop.set_postfix(loss=total_loss.item())

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for encoding, labels in val_loop:
                pixel_values = encoding["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

                outputs, enh_imgs, A = model(pixel_values=pixel_values, labels=labels)
                v_loss = outputs.loss + (args.enhancement_weight * (L_color(enh_imgs) + L_exp(enh_imgs) + L_spa(pixel_values, enh_imgs)))
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        experiment.log_metric("val_avg_loss", avg_val_loss, epoch=epoch)
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoints
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"joint_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()


# use class weight