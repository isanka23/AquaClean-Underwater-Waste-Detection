import json

def merge_coco_jsons(old_json_path, new_json_path, output_path):
    # 1. JSON ගොනු දෙක load කිරීම
    with open(old_json_path, 'r') as f:
        old_data = json.load(f)
    with open(new_json_path, 'r') as f:
        new_data = json.load(f)


    categories = [
        {"id": 0, "name": "plastic"},
        {"id": 1, "name": "rov"},
        {"id": 2, "name": "bio"}
    ]

    merged_images = old_data['images']
    merged_annotations = old_data['annotations']

    # දැනට තියෙන උපරිම IDs සොයා ගැනීම
    max_img_id = max([img['id'] for img in merged_images]) if merged_images else 0
    max_ann_id = max([ann['id'] for ann in merged_annotations]) if merged_annotations else 0

    # 2. අලුත් SeaClear දත්ත එකතු කිරීම (ID ගැටුම් වළක්වමින්)
    img_id_map = {}
    for img in new_data['images']:
        old_id = img['id']
        max_img_id += 1
        img_id_map[old_id] = max_img_id
        
        # Image object එක update කිරීම
        new_img = img.copy()
        new_img['id'] = max_img_id
        merged_images.append(new_img)

    for ann in new_data['annotations']:
        max_ann_id += 1
        new_ann = ann.copy()
        new_ann['id'] = max_ann_id
        new_ann['image_id'] = img_id_map[ann['image_id']]
        merged_annotations.append(new_ann)

    # 3. අවසාන JSON එක සකස් කිරීම
    final_data = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(final_data, f)
    
    print(f"✅ සාර්ථකයි! මුළු පින්තූර ගණන: {len(merged_images)}")

# Script එක run කිරීම
merge_coco_jsons('augmented_instances.json', 'seaclear_train.json', 'final_train_merged.json')