#Cell 4: The Ultimate GAN Training Loop This is the main engine. It handles the 90/10 dataset split, auto-resuming, PSNR evaluation, Comet ML logging, CSV writing, and Early Stopping.
# --- CELL 5: MAIN TRAINING ENGINE ---
import os
import csv
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# 1. System Check to ensure your files compiled properly
if not os.path.exists("model.py") or not os.path.exists("loss.py"):
    raise FileNotFoundError("❌ CRITICAL ERROR: model.py or loss.py is missing! Rerun Cells 3 and 4.")
print("✅ SYSTEM CHECK: Custom model and loss architectures found!")

from model import UNetGenerator, Discriminator
from loss import GANLoss

# --- COMET ML SETUP ---
from comet_ml import Experiment
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

experiment = Experiment(
    api_key="P1J4eRFU4Hx90OAG8Da7Ci9GW", 
    project_name="FYP-underwater_image_enhan"
)
experiment.set_name("Attention_GAN_Run_01")

# --- CONFIGURATION ---
def find_paired_folder(root_path):
    for root, dirs, files in os.walk(root_path):
        if 'Paired' in dirs: return root
    return "/content/dataset_local/EUVP_Dataset"

DATASET_ROOT = find_paired_folder("/content/dataset_local")
OUTPUT_DIR = "/content/drive/MyDrive/underwater_image_enhancment/AquaClean_GAN_Models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 8
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

csv_log_path = os.path.join(OUTPUT_DIR, "gan_training_metrics.csv")
latest_ckpt_path = os.path.join(OUTPUT_DIR, "gan_latest_checkpoint.pth")
best_ckpt_path = os.path.join(OUTPUT_DIR, "gan_best_checkpoint.pth")

print(f"✅ SYSTEM CHECK: Hardware initialized. Running on: {DEVICE.upper()} (Optimized for L4)")

# --- DATASET CLASS & AUTOMATIC 90/10 SPLIT ---
class GANDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths_a, self.image_paths_b = [], []
        targets = ['Paired/underwater_dark', 'Paired/underwater_imagenet', 'Paired/underwater_scenes']

        for t in targets:
            f_a = os.path.join(root_dir, t, 'trainA')
            f_b = os.path.join(root_dir, t, 'trainB')

            if os.path.exists(f_a) and os.path.exists(f_b):
                files_a = sorted(os.listdir(f_a))
                files_b = sorted(os.listdir(f_b))
                self.image_paths_a += [os.path.join(f_a, f) for f in files_a]
                self.image_paths_b += [os.path.join(f_b, f) for f in files_b]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.image_paths_a)
    def __getitem__(self, idx):
        img_a = self.transform(Image.open(self.image_paths_a[idx]).convert("RGB"))
        img_b = self.transform(Image.open(self.image_paths_b[idx]).convert("RGB"))
        return img_a, img_b

full_dataset = GANDataset(DATASET_ROOT)
val_size = int(0.10 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ SYSTEM CHECK: Found {len(full_dataset)} total images. Splitting into Train ({len(train_dataset)}) and Val ({len(val_dataset)})")

# --- INITIALIZE MODELS & OPTIMIZERS ---
generator = UNetGenerator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
criterion = GANLoss(DEVICE)

opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# --- THE FIX IS RIGHT HERE ---
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# --- AUTO-RESUME LOGIC ---
start_epoch = 0
best_psnr = 0.0
epochs_without_improvement = 0

if os.path.exists(latest_ckpt_path):
    print(f"\n✅ SUCCESS: Found existing checkpoint in Drive! Auto-resuming...")
    checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)

    generator.load_state_dict(checkpoint['gen_state'])
    discriminator.load_state_dict(checkpoint['disc_state'])
    opt_G.load_state_dict(checkpoint['opt_g_state'])
    opt_D.load_state_dict(checkpoint['opt_d_state'])

    start_epoch = checkpoint['epoch']
    best_psnr = checkpoint.get('best_psnr', 0.0)
    print(f"🔄 Resuming at Epoch {start_epoch + 1} | Best PSNR so far: {best_psnr:.2f} dB")
else:
    print("\n🚀 STARTING FRESH GAN TRAINING: No previous weights found. Creating log files...")
    with open(csv_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss_G', 'Loss_D', 'Val_PSNR', 'Val_SSIM', 'Time_Mins'])

print("\n" + "="*50)
print("🚀 COMMENCING TRAINING LOOP")
print("="*50 + "\n")

# --- MAIN TRAINING LOOP ---
for epoch in range(start_epoch, EPOCHS):
    epoch_start_time = time.time()

    # --- TRAIN PHASE ---
    generator.train()
    discriminator.train()
    g_losses, d_losses = [], []

    for i, (real_A, real_B) in enumerate(train_loader):
        real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

        # Train Discriminator
        opt_D.zero_grad()
        fake_B = generator(real_A)
        pred_real = discriminator(real_A, real_B)
        pred_fake = discriminator(real_A, fake_B.detach())
        loss_D = criterion.forward_D(pred_real, pred_fake)
        loss_D.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        pred_fake_G = discriminator(real_A, fake_B)
        loss_G = criterion.forward_G(fake_B, real_B, pred_fake_G)
        loss_G.backward()
        opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

    avg_g_loss = np.mean(g_losses)
    avg_d_loss = np.mean(d_losses)

    # --- VALIDATION PHASE ---
    generator.eval()
    val_psnr, val_ssim = 0.0, 0.0

    with torch.no_grad():
        for real_A, real_B in val_loader:
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
            fake_B = generator(real_A)

            # Normalize from [-1, 1] back to [0, 1] for accurate metrics
            fake_B_norm = (fake_B + 1.0) / 2.0
            real_B_norm = (real_B + 1.0) / 2.0

            val_psnr += psnr_metric(fake_B_norm, real_B_norm).item()
            val_ssim += ssim_metric(fake_B_norm, real_B_norm).item()

    avg_psnr = val_psnr / len(val_loader)
    avg_ssim = val_ssim / len(val_loader)

    epoch_time = (time.time() - epoch_start_time) / 60
    print(f"Epoch [{epoch+1}/{EPOCHS}] | G_Loss: {avg_g_loss:.3f} | D_Loss: {avg_d_loss:.3f} | PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.3f} | Time: {epoch_time:.1f}m")

    # --- LOGGING ---
    experiment.log_metrics({"Loss_G": avg_g_loss, "Loss_D": avg_d_loss, "Val_PSNR": avg_psnr, "Val_SSIM": avg_ssim, "Epoch_Time": epoch_time}, step=epoch+1)

    with open(csv_log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, avg_g_loss, avg_d_loss, avg_psnr, avg_ssim, round(epoch_time, 2)])

    # --- SAVING & EARLY STOPPING ---
    checkpoint_state = {
        'epoch': epoch + 1,
        'gen_state': generator.state_dict(),
        'disc_state': discriminator.state_dict(),
        'opt_g_state': opt_G.state_dict(),
        'opt_d_state': opt_D.state_dict(),
        'best_psnr': best_psnr
    }

    torch.save(checkpoint_state, latest_ckpt_path) # Save Last

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        checkpoint_state['best_psnr'] = best_psnr
        torch.save(checkpoint_state, best_ckpt_path) # Save Best
        print(f" -> 🌟 NEW BEST METRICS LOGGED! Saved best model weights.")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 5 == 0:
        torch.save(checkpoint_state, os.path.join(OUTPUT_DIR, f"gan_epoch_{epoch+1}.pth"))
        print(f" -> 💾 Interval Epoch {epoch+1} weights permanently saved to Drive.")

    if epochs_without_improvement >= PATIENCE:
        print(f"\n⚠️ EARLY STOPPING TRIGGERED! ⚠️")
        print(f"PSNR hasn't improved for {PATIENCE} consecutive epochs.")
        print("The GAN is likely starting to overfit or oversaturate images. Training stopped to preserve quality.")
        break

experiment.end()
print(f"\n✅ MISSION COMPLETE! Final Best Validation PSNR achieved: {best_psnr:.2f} dB")
