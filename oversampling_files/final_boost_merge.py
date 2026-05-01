import json

def final_merge(base_json_path, boost_json_path, output_path):
    with open(base_json_path, 'r') as f:
        base_data = json.load(f)
    with open(boost_json_path, 'r') as f:
        boost_data = json.load(f)

    merged_images = base_data['images']
    merged_annotations = base_data['annotations']
    categories = base_data['categories']

    max_img_id = max([img['id'] for img in merged_images]) if merged_images else 0
    max_ann_id = max([ann['id'] for ann in merged_annotations]) if merged_annotations else 0

    img_id_map = {}
    for img in boost_data['images']:
        old_id = img['id']
        max_img_id += 1
        img_id_map[old_id] = max_img_id
        
        new_img = img.copy()
        new_img['id'] = max_img_id
        new_img['file_name'] = "seaclear_augmented_images/" + img['file_name']
        merged_images.append(new_img)

    for ann in boost_data['annotations']:
        max_ann_id += 1
        new_ann = ann.copy()
        new_ann['id'] = max_ann_id
        new_ann['image_id'] = img_id_map[ann['image_id']]
        merged_annotations.append(new_ann)

    final_train_data = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(final_train_data, f)
    
    print(f"✅ Final merged dataset created.")
    print(f" {len(merged_images)}")
    print(f" {output_path} ")

final_merge('final_train_merged.json', 'seaclear_augmented_instances.json', 'final_train_with_rov_boost.json')