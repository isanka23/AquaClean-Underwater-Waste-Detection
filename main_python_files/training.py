import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
import os
from PIL import Image
from tqdm import tqdm
from models import JointModel

# ==========================================
# 1. ZERO-DCE LOSS FUNCTIONS (The Rules)
# ==========================================

#(Color Constancy) This loss calculates the difference between the R, G, and B channels and forces the model to balance them so the colors look natural.
def L_color(x):   
    mean_rgb = torch.mean(x, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    Drg = torch.pow(mr - mg, 2)
    Drb = torch.pow(mr - mb, 2)
    Dgb = torch.pow(mb - mg, 2)
    return torch.mean(torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5))

#(Exposure Control) This ensures the image isn't too dark or too bright. It compares the local average brightness to a mean_val of 0.6.
def L_exp(x, mean_val=0.6):
    x = torch.mean(x, 1, keepdim=True)
    avg = F.avg_pool2d(x, 16)
    return torch.mean(torch.pow(avg - mean_val, 2))

#(Spatial Consistency) find edges in the original image and ensures those same edges exist in the enhanced image.
def L_spa(org, enh):
    kernel = torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).expand(1, 1, 3, 3).to(org.device)
    org_mean = torch.mean(org, 1, keepdim=True)
    enh_mean = torch.mean(enh, 1, keepdim=True)
    org_grad = F.conv2d(org_mean, kernel, padding=1)
    enh_grad = F.conv2d(enh_mean, kernel, padding=1)
    return torch.mean(torch.pow(org_grad - enh_grad, 2))


# ==========================================
# 2. DATA LOADING
# ==========================================
class UnderwaterDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, processor):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        # 🟢 ADD THE LINE HERE TO TEST QUICKLY
        self.ids = self.ids[:100]
        self.img_folder = img_folder
        self.processor = processor

    # prepares each individual image and its labels before they are fed into the neural network.
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        
        # It needs a dict with 'image_id' and the list of 'annotations'
        formatted_annotations = {'image_id': img_id, 'annotations': target}
        
        encoding = self.processor(images=image, annotations=formatted_annotations, return_tensors="pt")
        
        # Remove the batch dimension added by the processor
        pixel_values = encoding["pixel_values"].squeeze(0) 
        labels = encoding["labels"][0]
        
        return pixel_values, labels

    def __len__(self):
        return len(self.ids)
    
# function stacks batch into a single mathematical tensor.
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = {"pixel_values": torch.stack(pixel_values)}
    labels = [item[1] for item in batch]
    return encoding, labels

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_img_dir = r"D:\4th year\fyp\AquaClean_Project\trash_ICRA19\trash_ICRA19\dataset\train"
    train_ann_file = "instances_train.json"

    # Init
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",
        size={"height": 800, "width": 800} # Forces a square resize for all images                                           
    )
    model = JointModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Lower LR for stability

    dataset = UnderwaterDataset(train_img_dir, train_ann_file, processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model.train()
    
    for epoch in range(60):
        loop = tqdm(loader, leave=True)
        epoch_loss = 0
        
        for batch_idx, (encoding, labels) in enumerate(loop):
            pixel_values = encoding["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            # Forward
            outputs, enhanced_imgs, A = model(pixel_values=pixel_values, labels=labels)
            
            # 1. Detection Loss
            det_loss = outputs.loss
            
            # 2. Enhancement Loss
            en_loss = L_color(enhanced_imgs) + L_exp(enhanced_imgs) + L_spa(pixel_values, enhanced_imgs)
            
            # Total Combined Loss
            total_loss = det_loss + (0.1 * en_loss)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            loop.set_description(f"Epoch [{epoch+1}/60]")
            loop.set_postfix(loss=total_loss.item())

        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss/len(loader):.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"joint_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()