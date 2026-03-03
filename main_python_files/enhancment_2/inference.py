import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Import your novel Generator with Attention
from model import UNetGenerator

def test_enhancement(image_path, model_path):
    # Set device to CPU since you are testing on your laptop
    device = torch.device('cpu')
    
    # 1. Initialize and Load the Model
    model = UNetGenerator().to(device)
    # Load the "Best" weights you just saved during training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # 2. Pre-processing (Match your training resolution)
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Matches your efficient training size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
    ])

    # Load the murky input
    input_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    # 3. Run Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 4. De-Normalize for Visualization
    # Convert [-1, 1] back to [0, 1]
    output_tensor = output_tensor.squeeze().cpu()
    output_tensor = output_tensor * 0.5 + 0.5 
    enhanced_img = transforms.ToPILImage()(output_tensor)

    # 5. Show Results Side-by-Side
    plt.figure(figsize=(12, 6))
    
    # Original Plot
    plt.subplot(1, 2, 1)
    plt.imshow(input_img.resize((128, 128)))
    plt.title("Original Murky Image")
    plt.axis("off")

    # Enhanced Plot
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img)
    plt.title("Attention-GAN Enhanced")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    MODEL_PATH = r"D:\4th year\fyp\AquaClean_Project\main_python_files\enhancment_2\results_gan\attention_gan_best.pth"
    # Pick any image from your trainA folder to see how it looks!
    TEST_IMAGE = r"D:\4th year\fyp\AquaClean_Project\test_images\270750_00035122.jpg"
    
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_IMAGE):
        test_enhancement(TEST_IMAGE, MODEL_PATH)
    else:
        print("Check your file paths! One of them does not exist.")