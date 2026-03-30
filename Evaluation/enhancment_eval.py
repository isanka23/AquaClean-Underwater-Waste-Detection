import os
import torch
import torch.nn as nn
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# ==========================================================
# 1. CONFIGURATION 
# ==========================================================
POOR_IMG_DIR = r"D:\4th year\fyp\under water waste detection\image enhancement\EUVP\Paired\underwater_imagenet\trainA" 

GOOD_IMG_DIR = r"D:\4th year\fyp\under water waste detection\image enhancement\EUVP\Paired\underwater_imagenet\trainB"

MODEL_PATH = r"D:\4th year\fyp\AquaClean_Project\enhancment\results_gan\gan_best_checkpoint.pth"

NUM_SAMPLES = 1000 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 2.  ARCHITECTURE 
# ==========================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        
        def conv_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize: layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def dec_block(in_ch, out_ch, dropout=False):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                      nn.InstanceNorm2d(out_ch),
                      nn.ReLU(inplace=True)]
            if dropout: layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.e1 = conv_block(in_channels, features, normalize=False)
        self.e2 = conv_block(features, features * 2)
        self.e3 = conv_block(features * 2, features * 4)
        self.e4 = conv_block(features * 4, features * 8)
        self.e5 = conv_block(features * 8, features * 8)
        self.attention = CBAM(features * 8)
        self.d1 = dec_block(features * 8, features * 8, dropout=True)
        self.d2 = dec_block(features * 8 * 2, features * 4, dropout=True)
        self.d3 = dec_block(features * 4 * 2, features * 2)
        self.d4 = dec_block(features * 2 * 2, features)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, 4, 2, 1), 
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.e1(x); enc2 = self.e2(enc1); enc3 = self.e3(enc2)
        enc4 = self.e4(enc3); enc5 = self.e5(enc4)
        focused = self.attention(enc5)
        dec1 = self.d1(focused)
        dec2 = self.d2(torch.cat([dec1, enc4], 1))
        dec3 = self.d3(torch.cat([dec2, enc3], 1))
        dec4 = self.d4(torch.cat([dec3, enc2], 1))
        return self.final(torch.cat([dec4, enc1], 1))

# ==========================================================
# 3. EVALUATION LOGIC
# ==========================================================
def run_benchmark():
    # Model Loading
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = UNetGenerator().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'gen_state' in checkpoint: 
        model.load_state_dict(checkpoint['gen_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Get and Sample Images
    all_images = [f for f in os.listdir(POOR_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    sampled_images = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    # Metrics & Transforms
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    psnr_scores, ssim_scores = [], []

    print(f"🚀 Benchmarking {len(sampled_images)} images on {DEVICE}...")

    with torch.no_grad():
        for img_file in tqdm(sampled_images):
            poor_path = os.path.join(POOR_IMG_DIR, img_file)
            gt_path = os.path.join(GOOD_IMG_DIR, img_file)

            if not os.path.exists(gt_path): continue

            # Load & Process Input
            raw_img = Image.open(poor_path).convert("RGB")
            input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

            # Load & Process Ground Truth
            gt_img = Image.open(gt_path).convert("RGB")
            gt_tensor = transforms.ToTensor()(transforms.Resize((256, 256))(gt_img)).unsqueeze(0).to(DEVICE)

            # Inference
            enhanced = model(input_tensor)
            # Tanh [-1, 1] -> [0, 1] normalization
            enhanced_norm = (enhanced + 1.0) / 2.0

            # Calculate Metrics
            psnr_scores.append(psnr_metric(enhanced_norm, gt_tensor).item())
            ssim_scores.append(ssim_metric(enhanced_norm, gt_tensor).item())

    # Results
    print("\n" + "="*35)
    print(f"📊 FINAL RESULTS (n={len(psnr_scores)})")
    print("-" * 35)
    print(f"✨ Avg PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"✨ Avg SSIM: {np.mean(ssim_scores):.4f}")
    print("="*35)

if __name__ == "__main__":
    run_benchmark()