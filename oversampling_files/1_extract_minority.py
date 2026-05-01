import os
import json
import cv2
import glob

# ==========================================================
# 1. CONFIGURATION 
# ==========================================================

IMG_ROOT_DIR = r"D:\4th year\fyp\obj_datasets\Seaclear Marine Debris Dataset" 

ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\SeaClear_data_process\seaclear_train.json" 

OUTPUT_DIR = r"seaclear_rov_crops" 


MINORITY_CLASSES = [1] 

# ==========================================================

# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading annotation file: {ANN_FILE}...")
with open(ANN_FILE, 'r') as f:
    coco = json.load(f)

img_dict = {img['id']: img['file_name'] for img in coco['images']}

print("Starting extraction process. Searching in sub-folders...")
count = 0

# ==========================================================
# ==========================================================

for ann in coco['annotations']:
    cat_id = ann['category_id']
    
    if cat_id in MINORITY_CLASSES:
        img_filename = img_dict[ann['image_id']]
        
        search_pattern = os.path.join(IMG_ROOT_DIR, "**", img_filename)
        found_files = glob.glob(search_pattern, recursive=True)

        if not found_files:
            print(f"⚠️ Image not found: {img_filename}")
            continue

        img_path = found_files[0] 
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        x, y, w, h = [int(v) for v in ann['bbox']]
        
        ih, iw, _ = image.shape
        x, y = max(0, x), max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)

        if w > 10 and h > 10:
            crop = image[y:y+h, x:x+w]
            
            class_name = "rov" 
            save_name = f"{class_name}_{count}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            cv2.imwrite(save_path, crop)
            count += 1
            
            if count % 100 == 0:
                print(f"Extracted {count} objects...")



print("-" * 30)
print(f"✅ Extraction complete!")
print(f"Saved {count} ROV crops to: {os.path.abspath(OUTPUT_DIR)}")
print("-" * 30)