import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import DetrImageProcessor, DetrForObjectDetection
from skimage.metrics import peak_signal_noise_ratio as psnr

import config
from metrics import calculate_uiqm

sys.path.append(config.MODEL_DIR)
try:
    from model import UNetGenerator
except ImportError as e:
    print(f"[ERROR] Could not find 'model.py' at {config.MODEL_DIR}. Details: {e}")
    raise ImportError(f"Could not find 'model.py' at {config.MODEL_DIR}")

class InferenceEngine:
    def __init__(self):
        self.device = config.DEVICE
        # loads the GAN and DETR models.
        self.load_models()

    def load_models(self):
        try:
            print("[INFO] Loading GAN Enhancement Model...")
            # 1. Load GAN
            self.gen = UNetGenerator().to(self.device)
            
            # --- THE SMART LOADER FIX FOR GAN ---
            gan_checkpoint = torch.load(config.GAN_WEIGHTS_PATH, map_location=self.device)
            
            # Check if it's our new dictionary format from cloud training
            if 'gen_state' in gan_checkpoint:
                self.gen.load_state_dict(gan_checkpoint['gen_state'])
                saved_epoch = gan_checkpoint.get('epoch', 'Unknown')
                print(f"[+] Loaded GAN dictionary model from Epoch {saved_epoch}")
            else:
                # Fallback for old raw weight files
                self.gen.load_state_dict(gan_checkpoint)
                print("[+] Loaded GAN model weights directly.")
                
            self.gen.eval()
            print("[INFO] GAN Loaded Successfully.")

            print("[INFO] Loading DETR Detection Model...")
            # 2. Load DETR
            self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detr_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=config.num_labels,
                id2label=config.id2label,
                label2id=config.label2id,
                ignore_mismatched_sizes=True,
            )
            
            print(f"[INFO] Fetching weights from: {config.DETR_WEIGHTS_PATH}")
            checkpoint = torch.load(config.DETR_WEIGHTS_PATH, map_location=self.device)
            
            # --- THE SMART LOADER FIX FOR DETR ---
            if 'model_state_dict' in checkpoint:
                self.detr_model.load_state_dict(checkpoint['model_state_dict'])
                saved_epoch = checkpoint.get('epoch', 'Unknown')
                print(f"[+] Loaded BEST dictionary model from Epoch {saved_epoch}")
            else:
                self.detr_model.load_state_dict(checkpoint)
                print("[+] Loaded interval model weights directly.")
                
            self.detr_model.to(self.device)
            self.detr_model.eval()
            print("[INFO] DETR Loaded Successfully.")
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to load AI models: {e}")
            raise e

    def process_image(self, img_path):
        try:
            print(f"[INFO] Starting inference on: {img_path}")
            start_time = time.time()
            
            # --- PREPARATION ---
            original_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_pil.size
            orig_np = np.array(original_pil).astype(np.float32) / 255.0
            input_uiqm_val = calculate_uiqm(orig_np)

            # --- STAGE 1: GAN ENHANCEMENT ---
            image_cv = cv2.imread(img_path)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            
            transform = A.Compose([
                A.Resize(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2(),
            ])
            input_tensor = transform(image=image_cv)["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                enhanced_tensor = self.gen(input_tensor)
            
            enh_tensor_sq = enhanced_tensor.squeeze().detach().cpu()
            enh_clamped = (enh_tensor_sq * 0.5 + 0.5).clamp(0, 1)
            enh_pil_256 = Image.fromarray((enh_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            enhanced_pil = enh_pil_256.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            
            enh_np = np.array(enhanced_pil).astype(np.float32) / 255.0
            enh_uiqm_val = calculate_uiqm(enh_np)
            psnr_val = psnr(orig_np, enh_np, data_range=1.0)

            # --- STAGE 2: DETECTION ---
            inputs = self.detr_processor(images=enhanced_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            target_sizes = torch.tensor([[orig_h, orig_w]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=config.CONFIDENCE_THRESHOLD
            )[0]

            # --- STAGE 3: POST-PROCESSING & DRAWING BOXES ---
            image_np = np.array(enhanced_pil).copy()
            detected_classes = set()
            count = 0
            total_score = 0.0 # Track score for accuracy calculation
            
            # 1. Calculate dynamic scaling factors based on image dimensions
            dynamic_scale = max(orig_w, orig_h) / 1000.0
            font_scale = max(0.5, dynamic_scale)  # Set a minimum font size
            text_thickness = max(2, int(font_scale * 2))
            bg_thickness = text_thickness + 2
            box_thickness = max(2, int(dynamic_scale * 3))
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > config.CONFIDENCE_THRESHOLD:
                    count += 1
                    total_score += score.item()
                    cls_name = config.id2label[label.item()]
                    detected_classes.add(cls_name)
                    box_color = config.BOX_COLORS.get(cls_name, (255, 255, 255)) 
                    
                    box = [int(i) for i in box.tolist()]
                    label_text = f"{cls_name}: {round(score.item(), 2)}"
                    
                    # 2. Draw the bounding box with dynamic thickness
                    cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), box_color, box_thickness)
                    
                    # 3. Draw text outline (black) with dynamic scale and thickness
                    cv2.putText(image_np, label_text, (box[0], box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), bg_thickness)
                    
                    # 4. Draw actual text (white) with dynamic scale and thickness
                    cv2.putText(image_np, label_text, (box[0], box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, config.TEXT_COLOR, text_thickness)

            result_pil = Image.fromarray(image_np)
            total_time = time.time() - start_time
            
            avg_accuracy = (total_score / count * 100) if count > 0 else 0.0
            
            # Print the mathematical metrics to the terminal only
            print("\n" + "="*40)
            print("📊 MATHEMATICAL METRICS (Terminal Only)")
            print(f"Input UIQM:    {input_uiqm_val:.3f}")
            print(f"Enhanced UIQM: {enh_uiqm_val:.3f}")
            print(f"PSNR:          {psnr_val:.2f} dB")
            print("="*40 + "\n")
            
            metrics = {
                "count": count,
                "classes": ", ".join(detected_classes) if count > 0 else "None",
                "accuracy": f"{avg_accuracy:.1f}%" if count > 0 else "N/A",
                "in_uiqm": round(input_uiqm_val, 3),
                "out_uiqm": round(enh_uiqm_val, 3),
                "psnr": round(psnr_val, 2),
                "time": round(total_time, 2)
            }
            
            print(f"[INFO] Inference successful. Found {count} objects.")
            return original_pil, result_pil, metrics
            
        except Exception as e:
            print(f"\n[ERROR] Inference failed during process_image: {e}")
            raise e