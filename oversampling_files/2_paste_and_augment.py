import os
import json
import cv2
import random
import glob

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
IMG_ROOT_DIR = r"D:\4th year\fyp\obj_datasets\Seaclear Marine Debris Dataset" 
ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\SeaClear_data_process\seaclear_train.json" 
CROPS_DIR = r"seaclear_rov_crops" 

AUG_IMG_DIR = r"seaclear_augmented_images"
AUG_ANN_FILE = r"seaclear_augmented_instances.json"

NUM_SYNTHETIC_IMAGES = 1500 

os.makedirs(AUG_IMG_DIR, exist_ok=True)

# ==========================================================
# 2. DATA LOADING
# ==========================================================
with open(ANN_FILE, 'r') as f:
    coco = json.load(f)

crop_files = glob.glob(os.path.join(CROPS_DIR, "*.jpg"))
print(f"Found {len(crop_files)} ROV crops to paste.")

def find_image_path(filename, root_dir):
    search_pattern = os.path.join(root_dir, "**", filename)
    found = glob.glob(search_pattern, recursive=True)
    return found[0] if found else None

next_ann_id = max([ann['id'] for ann in coco['annotations']]) + 1
next_img_id = max([img['id'] for img in coco['images']]) + 1

new_coco = {
    "images": [],
    "annotations": [],
    "categories": coco['categories']
}

# ==========================================================
# 3. GENERATION LOOP
# ==========================================================
print(f"Generating {NUM_SYNTHETIC_IMAGES} synthetic images...")

for i in range(NUM_SYNTHETIC_IMAGES):
    bg_info = random.choice(coco['images'])
    bg_path = find_image_path(bg_info['file_name'], IMG_ROOT_DIR)
    
    if bg_path is None: continue
    bg_img = cv2.imread(bg_path)
    if bg_img is None: continue

    crop_path = random.choice(crop_files)
    crop_img = cv2.imread(crop_path)
    
    bg_h, bg_w, _ = bg_img.shape
    crop_h, crop_w, _ = crop_img.shape

    if crop_h > bg_h or crop_w > bg_w:
        scale = min(bg_h/crop_h, bg_w/crop_w) * 0.5
        crop_img = cv2.resize(crop_img, (0,0), fx=scale, fy=scale)
        crop_h, crop_w, _ = crop_img.shape

    paste_x = random.randint(0, bg_w - crop_w)
    paste_y = random.randint(0, bg_h - crop_h)

    bg_img[paste_y:paste_y+crop_h, paste_x:paste_x+crop_w] = crop_img
    new_filename = f"rov_boost_{i}.jpg"
    cv2.imwrite(os.path.join(AUG_IMG_DIR, new_filename), bg_img)

    new_coco['images'].append({
        "id": next_img_id,
        "file_name": new_filename,
        "width": bg_w,
        "height": bg_h
    })

    new_coco['annotations'].append({
        "id": next_ann_id,
        "image_id": next_img_id,
        "category_id": 1, # ROV category
        "bbox": [paste_x, paste_y, crop_w, crop_h],
        "area": crop_w * crop_h,
        "iscrowd": 0
    })

    next_img_id += 1
    next_ann_id += 1
    if i % 100 == 0: print(f"Generated {i} images...")

with open(AUG_ANN_FILE, 'w') as f:
    json.dump(new_coco, f)

print(f"✅  {NUM_SYNTHETIC_IMAGES} ක් {AUG_IMG_DIR}  {AUG_ANN_FILE} ")