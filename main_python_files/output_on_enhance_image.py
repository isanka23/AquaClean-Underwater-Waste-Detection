import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor
import os

# ==========================================
# 1. RE-DEFINE THE CLASSES (Must match training)
# ==========================================
class DCENet(nn.Module):
    def __init__(self):
        super(DCENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
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
        enhanced_image, A = self.enhancer(pixel_values)
        outputs = self.detector(pixel_values=enhanced_image)
        return outputs, enhanced_image, A

# ==========================================
# 2. SETUP AND LOAD WEIGHTS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointModel().to(device)

# Load the weights you just finished training
model.load_state_dict(torch.load("joint_model_epoch_60.pt", map_location=device))
model.eval()

# Init Processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# ==========================================
# 3. RUN INFERENCE ON A TEST IMAGE
# ==========================================
img_path = r"D:\4th year\fyp\AquaClean_Project\test_images\obj1657_frame0000106.jpg" # Change this to a real image path
image = Image.open(img_path).convert("RGB")

# Prepare image
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs, enhanced_img, A = model(pixel_values=inputs['pixel_values'])

# Save the Enhanced image to see if Zero-DCE worked
enhanced_sample = enhanced_img[0].cpu().permute(1, 2, 0).numpy()
enhanced_sample = (enhanced_sample * 255).clip(0, 255).astype('uint8')
Image.fromarray(enhanced_sample).save("enhanced_visual_check.jpg")

# ==========================================
# 4. DRAW BOXES (Visualize Detection)
# ==========================================
# Keep only high-confidence detections (> 0.7)
probas = outputs.logits.softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.7

# Convert boxes to pixel coordinates
target_sizes = torch.tensor([image.size[::-1]]).to(device)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

draw = ImageDraw.Draw(image)
categories = ["plastic", "bio", "rov"]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{categories[label]}: {round(score.item(), 2)}", fill="white")

image.save("final_detection_result.jpg")
print("✅ Done! Check 'enhanced_visual_check.jpg' and 'final_detection_result.jpg'")