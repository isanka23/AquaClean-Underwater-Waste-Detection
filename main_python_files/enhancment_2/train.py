import os
import torch
import numpy as np
import time # --- NEW: Imported time module ---
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from model import UNetGenerator, Discriminator
from loss import GANLoss

class GANDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths_a, self.image_paths_b = [], []
        targets = ['Paired/underwater_dark', 'Paired/underwater_imagenet', 'Paired/underwater_scenes']
        for t in targets:
            f_a, f_b = os.path.join(root_dir, t, 'trainA'), os.path.join(root_dir, t, 'trainB')
            if os.path.exists(f_a):
                files = os.listdir(f_a)
                self.image_paths_a += [os.path.join(f_a, f) for f in files]
                self.image_paths_b += [os.path.join(f_b, f) for f in files]
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.image_paths_a)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths_a[idx]).convert('RGB')), \
               self.transform(Image.open(self.image_paths_b[idx]).convert('RGB'))

def train_gan(epochs=100):
    data_path = r"D:\4th year\fyp\under water waste detection\image enhancement\EUVP" 
    output_dir = r"D:\4th year\fyp\AquaClean_Project\main_python_files\enhancment_2\results_gan"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    criterion = GANLoss(device).to(device)
    
    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) 
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)) 

    full_ds = GANDataset(data_path)
    indices = np.random.permutation(len(full_ds))
    train_loader = DataLoader(Subset(full_ds, indices[:2500]), batch_size=1, shuffle=True) 
    
    # --- NEW: Updated Table Header ---
    print("-" * 75)
    print(f"{'Epoch':<5} | {'Time':<8} | {'G_Loss':<8} | {'D_Loss':<8} | {'Status'}")
    print("-" * 75)

    # --- NEW: Start Total Timer ---
    total_start_time = time.time()

    for epoch in range(epochs):
        # --- NEW: Start Epoch Timer ---
        epoch_start_time = time.time()
        
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []

        for i, (img_a, img_b) in enumerate(train_loader):
            img_a, img_b = img_a.to(device), img_b.to(device)
            
            fake_b = generator(img_a)
            
            # Train Discriminator
            opt_D.zero_grad()
            loss_D = criterion.forward_D(discriminator(img_a, img_b), discriminator(img_a, fake_b.detach()))
            loss_D.backward()
            opt_D.step()
            d_losses.append(loss_D.item())

            # Train Generator
            opt_G.zero_grad()
            loss_G = criterion.forward_G(fake_b, img_b, discriminator(img_a, fake_b))
            loss_G.backward()
            opt_G.step()
            g_losses.append(loss_G.item())

            if (i + 1) % 100 == 0:
                # Clear the line so the final print statement looks clean
                print(f" Batch {i+1}/2500", end="\r")

        # --- NEW: Calculate Epoch Time ---
        epoch_time = time.time() - epoch_start_time
        mins, secs = divmod(epoch_time, 60)
        time_str = f"{int(mins)}m {int(secs)}s"

        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        
        status = ""
        
        # Save a specific checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_name = f"attention_gan_epoch_{epoch+1}.pth"
            torch.save(generator.state_dict(), os.path.join(output_dir, checkpoint_name))
            status = "💾 SAVED CHECKPOINT"
            
        torch.save(generator.state_dict(), os.path.join(output_dir, "attention_gan_latest.pth"))

        # --- NEW: Print row with time ---
        print(f"{epoch+1:<5} | {time_str:<8} | {avg_g_loss:<8.4f} | {avg_d_loss:<8.4f} | {status}")

    # --- NEW: Calculate Total Training Time ---
    total_time = time.time() - total_start_time
    total_mins, total_secs = divmod(total_time, 60)
    total_hours, total_mins = divmod(total_mins, 60)
    
    print("-" * 75)
    print(f"🎉 Training Complete!")
    print(f"Total Time Taken: {int(total_hours)}h {int(total_mins)}m {int(total_secs)}s")
    print("-" * 75)

if __name__ == "__main__": 
    train_gan(epochs=100)