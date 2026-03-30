import json

# 1. SeaClear JSON එක load කරන්න
with open('dataset.json', 'r') as f:
    sc_data = json.load(f)

# 2. ඔබේ අලුත් Categories 3 නිර්මාණය කරන්න
NEW_CATEGORIES = [
    {"id": 0, "name": "plastic", "supercategory": "debris"},
    {"id": 1, "name": "rov", "supercategory": "robot"},
    {"id": 2, "name": "bio", "supercategory": "bio"}
]

# 3. Mapping Dictionary එක (SeaClear ID -> ඔබේ ID)
# PDF එකේ taxonomy අනුව IDs මෙතනට දාන්න [cite: 242]
mapping = {
    # Plastics (0)
    2: 0, 3: 0, 4: 0, 5: 0, 11: 0, 13: 0, 14: 0, 18: 0, 22: 0, 23: 0, 30: 0, 
    # ROV Parts (1)
    39: 1, 40: 1, # Robot/Robot parts [cite: 242]
    # Bio (2)
    6: 2, 8: 2, 9: 2, 12: 2, 15: 2, 19: 2, 21: 2, 26: 2, 29: 2 # Animals/Plants [cite: 239]
}

# 4. Annotations අලුත් IDs වලට update කිරීම
new_annotations = []
for ann in sc_data['annotations']:
    old_cat_id = ann['category_id']
    if old_cat_id in mapping:
        ann['category_id'] = mapping[old_cat_id]
        new_annotations.append(ann)

# 5. අදාළ පින්තූර පමණක් තෝරා ගැනීම
keep_image_ids = set([ann['image_id'] for ann in new_annotations])
new_images = [img for img in sc_data['images'] if img['id'] in keep_image_ids]

# 6. අවසාන JSON එක save කිරීම
mapped_data = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": NEW_CATEGORIES
}

with open('seaclear_mapped.json', 'w') as f:
    json.dump(mapped_data, f)

print(f"✅ Mapping is successful! Images processed: {len(new_images)}")