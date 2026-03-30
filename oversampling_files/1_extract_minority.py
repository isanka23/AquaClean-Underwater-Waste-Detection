# import os
# import json
# import cv2

# # --- CONFIGURATION ---
# IMG_DIR = r"D:\4th year\fyp\trash_ICRA19\trash_ICRA19\dataset\train"
# ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\coco_ano_files\instances_train.json"
# OUTPUT_DIR = r"D:\4th year\fyp\AquaClean_Project\extracted_minorities"

# # IDs based on your dictionary: {0: 'plastic', 1: 'bio', 2: 'rov'}
# MINORITY_CLASSES = [1, 2] 

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# with open(ANN_FILE, 'r') as f:
#     coco = json.load(f)

# # Create a lookup dictionary for images
# img_dict = {img['id']: img['file_name'] for img in coco['images']}

# print("Extracting ROV and BIO objects...")
# count = 0

# for ann in coco['annotations']:
#     cat_id = ann['category_id']
#     if cat_id in MINORITY_CLASSES:
#         img_filename = img_dict[ann['image_id']]
#         img_path = os.path.join(IMG_DIR, img_filename)
        
#         image = cv2.imread(img_path)
#         if image is None: continue

#         # Get bounding box [x, y, width, height]
#         x, y, w, h = [int(v) for v in ann['bbox']]
        
#         # Ensure coordinates are within image boundaries
#         ih, iw, _ = image.shape
#         x, y = max(0, x), max(0, y)
#         w = min(w, iw - x)
#         h = min(h, ih - y)

#         if w > 10 and h > 10: # Only save if it's a reasonably sized crop
#             crop = image[y:y+h, x:x+w]
#             class_name = "bio" if cat_id == 1 else "rov"
#             save_name = f"{class_name}_{count}.jpg"
#             cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), crop)
#             count += 1

# print(f"Extraction complete! Saved {count} minority objects to {OUTPUT_DIR}")



import os
import json
import cv2
import glob

# ==========================================================
# 1. CONFIGURATION (පරිශීලක සැකසුම්)
# ==========================================================

# SeaClear පින්තූර තියෙන ප්‍රධාන (Root) folder එකේ path එක
IMG_ROOT_DIR = r"D:\4th year\fyp\obj_datasets\Seaclear Marine Debris Dataset" 

# අපි කලින් හදාගත්ත seaclear_train.json file එකේ path එක
ANN_FILE = r"D:\4th year\fyp\AquaClean_Project\SeaClear_data_process\seaclear_train.json" 

# කපාගන්නා ROV කොටස් save කරන folder එකේ නම
OUTPUT_DIR = r"seaclear_rov_crops" 

# අපේ Mapping එකට අනුව ROV සඳහා ID එක 1 වේ
MINORITY_CLASSES = [1] 

# ==========================================================
# 2. සූදානම් කිරීම්
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading annotation file: {ANN_FILE}...")
with open(ANN_FILE, 'r') as f:
    coco = json.load(f)

# පින්තූර වල IDs සහ නම් එකිනෙකට ගැලපීමට dictionary එකක් සෑදීම
img_dict = {img['id']: img['file_name'] for img in coco['images']}

print("Starting extraction process. Searching in sub-folders...")
count = 0

# ==========================================================
# 3. ප්‍රධාන ක්‍රියාවලිය (Extraction Loop)
# ==========================================================

for ann in coco['annotations']:
    cat_id = ann['category_id']
    
    # අපට අවශ්‍ය ROV (ID 1) පමණක් නම්
    if cat_id in MINORITY_CLASSES:
        img_filename = img_dict[ann['image_id']]
        
        # --- පින්තූරය sub-folders ඇතුළේ සෙවීම (Recursive Search) ---
        search_pattern = os.path.join(IMG_ROOT_DIR, "**", img_filename)
        found_files = glob.glob(search_pattern, recursive=True)

        if not found_files:
            print(f"⚠️ පින්තූරය හමුවුණේ නැහැ: {img_filename}")
            continue

        img_path = found_files[0] # හමුවුණු පළමු පින්තූරයේ path එක ගැනීම
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"❌ පින්තූරය කියවිය නොහැක: {img_path}")
            continue

        # Bounding box එක ලබා ගැනීම [x, y, width, height]
        x, y, w, h = [int(v) for v in ann['bbox']]
        
        # පින්තූරයේ සීමාවන් තුළ Coordinates පවතින බව තහවුරු කිරීම
        ih, iw, _ = image.shape
        x, y = max(0, x), max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)

        # ඉතා කුඩා පින්තූර කැබලි මඟ හැරීම (pixel 10 ට වඩා විශාල විය යුතුය)
        if w > 10 and h > 10:
            crop = image[y:y+h, x:x+w]
            
            # පින්තූරය නම් කර save කිරීම
            class_name = "rov" # මක්නිසාද cat_id == 1
            save_name = f"{class_name}_{count}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            cv2.imwrite(save_path, crop)
            count += 1
            
            if count % 100 == 0:
                print(f"Extracted {count} objects...")



print("-" * 30)
print(f"✅ Extraction සාර්ථකව අවසන්!")
print(f"Saved {count} ROV crops to: {os.path.abspath(OUTPUT_DIR)}")
print("-" * 30)