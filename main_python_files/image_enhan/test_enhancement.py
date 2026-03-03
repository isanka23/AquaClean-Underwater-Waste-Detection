import os
import torch
from torchvision import transforms
from PIL import Image
from model import AquaCleanNet 
import matplotlib.pyplot as plt

def verify_best_model(test_image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path to the best model saved by your new train.py
    model_path = r"D:\4th year\fyp\AquaClean_Project\main_python_files\image_enhan\aquaclean_results\best_model.pth"
    
    model = AquaCleanNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    raw_img = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
    
    enhanced_img = transforms.ToPILImage()(enhanced_tensor.squeeze().cpu())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(raw_img.resize((256, 256))); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(enhanced_img); plt.title("AquaClean Best Model"); plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_img = r"D:\4th year\fyp\AquaClean_Project\test_images\bio0001_frame0000475.jpg"
    verify_best_model(test_img)