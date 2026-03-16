import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# --- THE FIX: Tell Python where to find model.py and metrics.py ---
# 1. Path to your model.py
sys.path.append(r"D:\4th year\fyp\AquaClean_Project\enhancment")
# 2. Path to your metrics.py (your main app folder)
sys.path.append(r"D:\4th year\fyp\AquaClean_Project\aqua_clean")

# Now it will successfully import both!
from model import UNetGenerator
from metrics import calculate_uiqm

def load_smart_model(weights_path, device):
    """Loads the UNetGenerator whether it's a raw weight file or a dictionary."""
    model = UNetGenerator().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Check if it's the Colab dictionary format
    if 'gen_state' in checkpoint:
        model.load_state_dict(checkpoint['gen_state'])
    else:
        # Fallback for the VS Code raw weights format
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def compare_models(image_paths, vs_code_model_path, colab_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device.type.upper()}")
    
    print("Loading VS Code Model...")
    model_vscode = load_smart_model(vs_code_model_path, device)
    
    print("Loading Colab Model...")
    model_colab = load_smart_model(colab_model_path, device)

    # Standard GAN transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Setup matplotlib grid
    num_imgs = len(image_paths)
    # Increased height slightly to accommodate two lines of title text
    fig, axes = plt.subplots(num_imgs, 3, figsize=(12, 4.5 * num_imgs))
    
    # Ensure axes is always a 2D array even if there is only 1 image
    if num_imgs == 1:
        axes = [axes] 

    print("Running Inference and Calculating Metrics...")
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            # 1. Load and prepare image
            orig_pil = Image.open(img_path).convert("RGB")
            input_tensor = transform(orig_pil).unsqueeze(0).to(device)

            # 2. Run both models
            out_vscode = model_vscode(input_tensor)
            out_colab = model_colab(input_tensor)

            # 3. Denormalize from [-1, 1] to [0, 1] for display and metrics
            orig_disp = orig_pil.resize((256, 256))
            orig_np = np.array(orig_disp).astype(np.float32) / 255.0  # Prepare original for UIQM
            
            vscode_disp = out_vscode.squeeze().cpu().permute(1, 2, 0).numpy()
            vscode_disp = (vscode_disp * 0.5 + 0.5).clip(0, 1)
            
            colab_disp = out_colab.squeeze().cpu().permute(1, 2, 0).numpy()
            colab_disp = (colab_disp * 0.5 + 0.5).clip(0, 1)

            # --- CALCULATE METRICS ---
            orig_uiqm = calculate_uiqm(orig_np)
            vscode_uiqm = calculate_uiqm(vscode_disp)
            colab_uiqm = calculate_uiqm(colab_disp)

            # 4. Plot Original
            axes[i][0].imshow(orig_disp)
            axes[i][0].set_title(f"Original Input\nUIQM: {orig_uiqm:.3f}", fontsize=14)
            axes[i][0].axis('off')

            # 5. Plot VS Code Result
            axes[i][1].imshow(vscode_disp)
            axes[i][1].set_title(f"Model 1 (VS Code)\nUIQM: {vscode_uiqm:.3f}", fontsize=14)
            axes[i][1].axis('off')

            # 6. Plot Colab Result
            axes[i][2].imshow(colab_disp)
            axes[i][2].set_title(f"Model 2 (Colab Best)\nUIQM: {colab_uiqm:.3f}", fontsize=14)
            axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# RUN THE TEST
# ==========================================
if __name__ == "__main__":
    # Put the paths to a few test images here
    test_images = [
        r"D:\4th year\fyp\AquaClean_Project\test_images\test_9060up.jpg",
        r"D:\4th year\fyp\AquaClean_Project\test_images\nm_358up.jpg",
        r"D:\4th year\fyp\AquaClean_Project\test_images\nm_321up.jpg",
        r"D:\4th year\fyp\AquaClean_Project\test_images\nm_313up.jpg"
    ]
    
    # Put the paths to your two model files here
    vscode_weights = r"D:\4th year\fyp\AquaClean_Project\enhancment\results_gan\attention_gan_epoch_100.pth"
    colab_weights = r"D:\4th year\fyp\AquaClean_Project\enhancment\results_gan\gan_best_checkpoint.pth"
    
    compare_models(test_images, vscode_weights, colab_weights)