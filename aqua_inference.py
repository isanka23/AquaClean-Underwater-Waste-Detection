import sys
import os

# --- TELL PYTHON WHERE TO FIND MODEL.PY ---
sys.path.append(r"D:\4th year\fyp\AquaClean_Project\main_python_files\enhancment_2") 

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import DetrImageProcessor, DetrForObjectDetection

# --- IMPORT YOUR GENERATOR MODEL ---
try:
    from model import UNetGenerator
except ImportError:
    raise ImportError("Could not import 'UNetGenerator' from 'model.py'.")

# =================CONFIGURATION=================
GAN_WEIGHTS_PATH = r'D:\4th year\fyp\AquaClean_Project\main_python_files\enhancment_2\results_gan\attention_gan_epoch_100.pth'
DETR_WEIGHTS_PATH = r'D:\4th year\fyp\AquaClean_Project\object_detecion\detr_trash_model_epoch_50.pth'
INPUT_IMAGE_PATH = r'D:\4th year\fyp\AquaClean_Project\test_images\nm_405up.jpg'

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.80

id2label = {0: 'plastic', 1: 'bio', 2: 'rov'} 
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
# ===============================================

def load_models():
    print("Loading models...")
    gen = UNetGenerator().to(DEVICE)
    gen.load_state_dict(torch.load(GAN_WEIGHTS_PATH, map_location=DEVICE))
    gen.eval() 

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(DETR_WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() 
    return gen, processor, model

def preprocess_for_gan(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return transform(image=image)["image"].unsqueeze(0).to(DEVICE)

def postprocess_gan_output(tensor):
    image = tensor.squeeze().detach().cpu()
    image = (image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255).astype(np.uint8))

def run_detection(processor, model, image, orig_size):
    """Encapsulated DETR inference and post-processing logic."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([orig_size]).to(DEVICE)
    return processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD)[0]

def visualize_detections(image, results):
    image_np = np.array(image).copy()
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > CONFIDENCE_THRESHOLD:
            box = [int(i) for i in box.tolist()]
            label_text = f"{id2label[label.item()]}: {round(score.item(), 2)}"
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            (w, h), b = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_np, (box[0], box[1] - h - b - 8), (box[0] + w, box[1]), (0, 255, 0), -1)
            cv2.putText(image_np, label_text, (box[0], box[1] - b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image_np

def main():
    gen, processor, detr_model = load_models()
    original_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    orig_w, orig_h = original_pil.size
    orig_size = (orig_h, orig_w)

    # --- STAGE 1: GAN ENHANCEMENT ---
    input_tensor = preprocess_for_gan(INPUT_IMAGE_PATH)
    with torch.no_grad():
        enhanced_tensor = gen(input_tensor)
    enhanced_pil = postprocess_gan_output(enhanced_tensor).resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # --- STAGE 2: DETECTION ON ORIGINAL ---
    print("Detecting on Original...")
    results_orig = run_detection(processor, detr_model, original_pil, orig_size)
    vis_orig = visualize_detections(original_pil, results_orig)

    # --- STAGE 3: DETECTION ON ENHANCED ---
    print("Detecting on Enhanced...")
    results_enh = run_detection(processor, detr_model, enhanced_pil, orig_size)
    vis_enh = visualize_detections(enhanced_pil, results_enh)

    # --- FINAL VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(vis_orig)
    axes[0].set_title(f"Detection on Original\nThreshold: {CONFIDENCE_THRESHOLD}", fontsize=15)
    axes[0].axis('off')

    axes[1].imshow(vis_enh)
    axes[1].set_title(f"Detection on GAN Enhanced\nThreshold: {CONFIDENCE_THRESHOLD}", fontsize=15)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()