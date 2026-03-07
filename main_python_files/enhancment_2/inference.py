import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import UNetGenerator
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- UIQM Calculation Functions ---
def calculate_uiqm(img):
    """Simplified UIQM implementation (Colorfulness, Sharpness, Contrast)"""
    # UIQM is complex; this is a standard approximation for research purposes
    img_gb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # 1. UICM (Colorfulness)
    rg = img_gb[:,:,2] - img_gb[:,:,1]
    yb = 0.5 * (img_gb[:,:,2] + img_gb[:,:,1]) - img_gb[:,:,0]
    uicm = 0.02 * np.sqrt(np.mean(rg**2) + np.mean(yb**2))
    
    # 2. UISM (Sharpness)
    gray = cv2.cvtColor(img_gb, cv2.COLOR_BGR2GRAY)
    uism = cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0 # Normalized variance
    
    # 3. UIConM (Contrast)
    uiconm = np.std(gray) / 255.0
    
    # Final Weighted UIQM
    return (0.3 * uicm) + (0.5 * uism) + (0.2 * uiconm)

def test_single_image(checkpoint_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    
    # 2. Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    with torch.no_grad():
        img_raw = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_raw).unsqueeze(0).to(device)
        fake_img = generator(img_tensor)
        
        # 3. Convert to Numpy for Metrics [0, 1]
        orig_np = ((img_tensor.squeeze(0).cpu() + 1.0) / 2.0).permute(1, 2, 0).numpy()
        enh_np = ((fake_img.squeeze(0).cpu() + 1.0) / 2.0).permute(1, 2, 0).numpy()
        
        # 4. Calculate Metrics
        # Note: PSNR/SSIM compare Enhanced vs. Original (assuming Original is the target)
        # In underwater, PSNR might be low even if the image looks 'better' to you.
        cur_psnr = psnr(orig_np, enh_np, data_range=1.0)
        cur_ssim = ssim(orig_np, enh_np, data_range=1.0, channel_axis=2)
        
        uiqm_orig = calculate_uiqm(orig_np)
        uiqm_enh = calculate_uiqm(enh_np)

        # 5. Plotting
        plt.figure(figsize=(12, 7))
        
        # Input Image
        plt.subplot(1, 2, 1)
        plt.imshow(orig_np)
        plt.title(f"Original Input\nUIQM: {uiqm_orig:.3f}")
        plt.axis('off')
        
        # Enhanced Image
        plt.subplot(1, 2, 2)
        plt.imshow(enh_np)
        plt.title(f"Enhanced (Epoch {os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]})\n"
                  f"PSNR: {cur_psnr:.2f} | SSIM: {cur_ssim:.3f}\n"
                  f"UIQM: {uiqm_enh:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    CHECKPOINT_FILE = r"D:\4th year\fyp\AquaClean_Project\main_python_files\enhancment_2\results_gan\attention_gan_epoch_100.pth"
    IMAGE_FILE = r"D:\4th year\fyp\AquaClean_Project\test_images\nm_124up.jpg" 
    
    test_single_image(CHECKPOINT_FILE, IMAGE_FILE)