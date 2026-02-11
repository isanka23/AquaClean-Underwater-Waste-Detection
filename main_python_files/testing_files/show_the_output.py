import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINE CLASSES
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
# 2. SETUP AND INFERENCE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointModel().to(device)
model.load_state_dict(torch.load("joint_model_epoch_60.pt", map_location=device))
model.eval()

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

img_path = r"D:\4th year\fyp\AquaClean_Project\trash_ICRA19\trash_ICRA19\dataset\test\bio0000_frame0000016.jpg"
original_image = Image.open(img_path).convert("RGB")
# We make a copy to draw boxes on specifically
detection_image = original_image.copy() 

inputs = processor(images=original_image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs, enhanced_img, A = model(pixel_values=inputs['pixel_values'])

# Convert Enhanced Tensor to Image
enhanced_np = enhanced_img[0].cpu().permute(1, 2, 0).numpy()
enhanced_np = (enhanced_np * 255).clip(0, 255).astype('uint8')
enhanced_pil = Image.fromarray(enhanced_np)

# ==========================================
# 3. PROCESS DETECTIONS (Draw on detection_image)
# ==========================================
target_sizes = torch.tensor([original_image.size[::-1]]).to(device)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

draw = ImageDraw.Draw(detection_image)
categories = ["plastic", "bio", "rov"]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    
    # 🟢 NEW: Format label and score together (e.g., "bio: 0.95")
    label_text = f"{categories[label]}: {round(score.item() * 100, 1)}%"
    
    draw.rectangle(box, outline="#00FF00", width=4) 
    
    # 2. Draw a small background for the text (makes red text easier to see)
    text_pos = (box[0], box[1] - 25)
    
    # 3. Draw the label in RED color
    draw.text(text_pos, label_text, fill="red")

# Update the plot title to show average confidence if you want
avg_conf = results["scores"].mean().item() if len(results["scores"]) > 0 else 0

# ==========================================
# 4. SHOW 3 PICTURES SIDE-BY-SIDE
# ==========================================
plt.figure(figsize=(20, 7))

# Plot 1: Original Murky Input
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("1. Original Input")
plt.axis('off')

# Plot 2: Enhanced (Restoration Result)
plt.subplot(1, 3, 2)
plt.imshow(enhanced_pil)
plt.title("2. Zero-DCE Enhanced (Clean)")
plt.axis('off')

# Plot 3: Final Detection
plt.subplot(1, 3, 3)
plt.imshow(detection_image)
plt.title(f"3. DETR Detections ({len(results['labels'])} found)")
plt.axis('off')

plt.tight_layout()
plt.show()