import os
from ultralytics import YOLO
import torch

# ==========================================================
# 1. පාරවල් (Paths)
# ==========================================================
DATASET_ROOT = r"D:\4th year\fyp\under water waste detection\trash_ICRA19\trash_ICRA19\dataset"

# ==========================================================
# 2. YAML FILE එක සැකසීම
# ==========================================================
yaml_content = f"""
path: {DATASET_ROOT.replace('\\', '/')}
train: train
val: val
test: test

names:
  0: plastic
  1: bio
  2: rov
"""
yaml_file = 'trash_yolo_val_comparison.yaml'
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

# ==========================================================
# 3. BENCHMARKING FUNCTION
# ==========================================================
def run_comparison():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # YOLO versions (v11 සඳහා නිවැරදි නම yolo11n.pt වේ)
    models_to_test = ['yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt']
    results_summary = []

    print(f"🚀 Starting Multi-Model VALIDATION Evaluation on: {DEVICE}")

    for model_name in models_to_test:
        print(f"\n" + "-"*30)
        print(f"🔄 Evaluating Model (Validation): {model_name}")
        print("-"*30)
        
        try:
            model = YOLO(model_name)

            # split='val' ලෙස වෙනස් කිරීමෙන් validation data පාවිච්චි වේ
            metrics = model.val(
                data=yaml_file,
                split='val',      # <--- මෙතැන 'test' වෙනුවට 'val' ලෙස මාරු කරන ලදී
                imgsz=640,
                device=DEVICE,
                verbose=False
            )

            # Metrics ලබා ගැනීම
            mAP50 = metrics.box.map50
            recall = metrics.box.mr
            precision = metrics.box.mp
            inference_time = metrics.speed['inference']

            fps = 1000 / inference_time if inference_time > 0 else 0

            results_summary.append({
                "Model": model_name,
                "mAP@50": mAP50,
                "Precision": precision,
                "Recall": recall,
                "FPS": fps
            })
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")

    # 4. අවසාන ප්‍රතිඵල Table එකක් ලෙස පෙන්වීම
    print("\n" + "="*85)
    print(f"📊 YOLO VALIDATION EVALUATION RESULTS")
    print("-" * 85)
    print(f"{'Model Name':<15} | {'mAP@50':<12} | {'Precision':<12} | {'Recall':<12} | {'FPS':<10}")
    print("-" * 85)
    for res in results_summary:
        print(f"{res['Model']:<15} | {res['mAP@50']:<12.4f} | {res['Precision']:<12.4f} | {res['Recall']:<12.4f} | {res['FPS']:<10.2f}")
    print("="*85)

if __name__ == "__main__":
    run_comparison()