import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from model import AquaCleanNet
from loss import UnderwaterLosses
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_metrics(enh, gt):
    enh_np = enh.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    gt_np = gt.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    p = psnr(gt_np, enh_np, data_range=1)
    s = ssim(gt_np, enh_np, data_range=1, channel_axis=2)
    u = (np.std(enh_np) + np.var(enh_np)) * 10 
    return p, s, u

class PairedEUVPDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths_a, self.image_paths_b = [], []
        targets = ['Paired/underwater_dark', 'Paired/underwater_imagenet', 'Paired/underwater_scenes']
        for t in targets:
            f_a, f_b = os.path.join(root_dir, t, 'trainA'), os.path.join(root_dir, t, 'trainB')
            if os.path.exists(f_a):
                files = os.listdir(f_a)
                self.image_paths_a += [os.path.join(f_a, f) for f in files]
                self.image_paths_b += [os.path.join(f_b, f) for f in files]
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    def __len__(self): return len(self.image_paths_a)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths_a[idx]).convert('RGB')), \
               self.transform(Image.open(self.image_paths_b[idx]).convert('RGB'))

def train_aquaclean(epochs=100):
    output_dir = r"D:\4th year\fyp\AquaClean_Project\main_python_files\image_enhan\aquaclean_results"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AquaCleanNet().to(device)
    criterion = UnderwaterLosses().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005) 

    full_ds = PairedEUVPDataset(r"D:\4th year\fyp\under water waste detection\image enhancement\EUVP")
    indices = np.random.permutation(len(full_ds))
    train_loader = DataLoader(Subset(full_ds, indices[:500]), batch_size=1, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, indices[500:550]), batch_size=1, shuffle=False)

    print(f"--- Perceptual-Guided Training (CPU) ---")
    print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 115)

    history, best_score = [], -float('inf') 

    print(f"{'Epoch':<5} | {'Started':<9} | {'Live/Total':<12} | {'Remaining':<9} | {'Loss':<8} | {'PSNR':<6} | {'SSIM':<6} | {'Status'}")
    print("-" * 115)

    for epoch in range(epochs):
        start_timestamp = datetime.now().strftime('%H:%M:%S')
        epoch_start_time = time.time()
        model.train()
        l_total = []
        
        for i, (img_a, img_b) in enumerate(train_loader):
            img_a, img_b = img_a.to(device), img_b.to(device)
            optimizer.zero_grad()
            enh = model(img_a)
            
            # Loss now compares against the ground truth (img_b)
            loss = criterion(img_a, enh, img_b)
            loss.backward()
            optimizer.step()
            l_total.append(loss.item())
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - epoch_start_time
                print(f"  > Epoch {epoch+1} Progress: {i+1}/500 images | Live Time: {int(elapsed//60)}m {int(elapsed%60)}s", end='\r')

        model.eval()
        v_psnr, v_ssim, v_uiqm = [], [], []
        with torch.no_grad():
            for img_a, img_b in val_loader:
                img_a, img_b = img_a.to(device), img_b.to(device)
                enh = model(img_a)
                p, s, u = get_metrics(enh, img_b)
                v_psnr.append(p); v_ssim.append(s); v_uiqm.append(u)

        epoch_duration = time.time() - epoch_start_time
        duration_str = f"{int(epoch_duration // 60)}m {int(epoch_duration % 60)}s"
        remaining_str = f"{int((epoch_duration * (epochs - (epoch + 1))) // 60)}m"
        
        avg_uiqm, avg_psnr, avg_ssim, avg_loss = np.mean(v_uiqm), np.mean(v_psnr), np.mean(v_ssim), np.mean(l_total)
        current_score = (avg_ssim * 100) + (avg_uiqm * 5) + (avg_psnr / 5.0) - (avg_loss * 2)
        
        status = ""
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            status = "⭐ SAVED BEST"

        print(f"{epoch+1:<5} | {start_timestamp:<9} | {duration_str:<12} | {remaining_str:<9} | {avg_loss:<8.4f} | {avg_psnr:<6.2f} | {avg_ssim:<6.3f} | {status}")
        history.append({"Epoch": epoch + 1, "Loss": avg_loss, "PSNR": avg_psnr, "SSIM": avg_ssim})

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "perceptual_training_log.csv"), index=False)

if __name__ == "__main__": train_aquaclean()