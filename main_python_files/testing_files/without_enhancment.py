import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor
import os

# ==========================================
# 1. MODEL DEFINITION (Keeping JointModel structure for loading weights)
# ==========================================

class DCENet(nn.Module):
    def __init__(self):
        super(DCENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        A = torch.tanh(self.conv3(x2))
        enhanced = x + A * x * (1 - x)
        return enhanced, A

class JointModel(nn.Module):
    def __init__(self, detector_model="facebook/detr-resnet-50"):
        super(JointModel, self).__init__()
        self.enhancer = DCENet()
        self.detector = DetrForObjectDetection.from_pretrained(
            detector_model, num_labels=3, ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        # We define this to match the saved state_dict structure
        enhanced_image, A = self.enhancer(pixel_values)
        outputs = self.detector(pixel_values=enhanced_image)
        return outputs, enhanced_image, A
def run_raw_detection(img_path, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created folder: {output_dir}")

    # 2. Load Model
    model = JointModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    categories = ["plastic", "bio", "rov"]

    # 3. Process Image
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)

    # 4. Detect (Bypassing Enhancer)
    with torch.no_grad():
        outputs = model.detector(pixel_values=inputs['pixel_values'])
    
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # 5. Draw
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()
        draw.rectangle(box, outline="red", width=4)
        text = f"{categories[label]}: {round(score.item(), 2)}"
        draw.text((box[0], box[1] - 15), text, fill="white")

    # 6. Save to specific path
    save_path = os.path.join(output_dir, "raw_detection_result.jpg")
    image.save(save_path)
    
    print(f"✅ Success! Image stored at: {save_path}")

if __name__ == "__main__":
    # YOUR SPECIFIC PATHS
    TEST_IMAGE = r"D:\4th year\fyp\AquaClean_Project\trash_ICRA19\trash_ICRA19\dataset\test\bio0004_frame0000161.jpg"
    MODEL_WEIGHTS = r"D:\4th year\fyp\AquaClean_Project\epoch_files\joint_model_epoch_60.pt"
    OUTPUT_FOLDER = r"D:\4th year\fyp\AquaClean_Project\output_images"

    run_raw_detection(TEST_IMAGE, MODEL_WEIGHTS, OUTPUT_FOLDER)