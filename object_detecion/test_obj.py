import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt


MODEL_PATH = r"D:\4th year\fyp\AquaClean_Project\object_detecion\test_weights\detr_best++.pth"
IMAGE_PATH = r"D:\4th year\fyp\AquaClean_Project\test_images\nm_321up.jpg"
CONFIDENCE_THRESHOLD = 0.3  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD MODEL & PROCESSOR ---
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,
    id2label={0: "plastic", 1: "rov", 2: "bio"},
    ignore_mismatched_sizes=True
).to(device)


if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Weights loaded successfully from: {os.path.basename(MODEL_PATH)}")
else:
    print("❌ ERROR: Model weights not found! Please check the path.")

# --- 3. INFERENCE FUNCTION ---
def run_detection(img_path, threshold):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)


    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    draw = ImageDraw.Draw(image)
    

    colors = {"plastic": "red", "rov": "blue", "bio": "green"}

    print(f"🔍 Found {len(results['scores'])} objects:")
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        
  
        draw.rectangle(box, outline=colors.get(label_name, "white"), width=4)
        draw.text((box[0], box[1]), f"{label_name}: {score:.2f}", fill="white")
        
        print(f" -> Label: {label_name} | Confidence: {score:.2f} | Box: {box}")

    return image

# --- 4. EXECUTE & DISPLAY ---
result_img = run_detection(IMAGE_PATH, CONFIDENCE_THRESHOLD)

plt.figure(figsize=(12, 8))
plt.imshow(result_img)
plt.axis('off')
plt.show()