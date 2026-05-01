import json
import random
import os

with open('seaclear_mapped.json', 'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']


scenes = {}
for img in images:
    # filename එකේ අවසාන කොටස (timestamp) scene ID එක විදිහට ගන්නවා
    scene_id = os.path.basename(img['file_name']).split('.')[0]
    if scene_id not in scenes:
        scenes[scene_id] = []
    scenes[scene_id].append(img)

scene_keys = list(scenes.keys())
random.seed(42)
random.shuffle(scene_keys)

split_idx = int(len(scene_keys) * 0.8)
train_scenes = scene_keys[:split_idx]
test_scenes = scene_keys[split_idx:]

train_images = []
for s in train_scenes: train_images.extend(scenes[s])

test_images = []
for s in test_scenes: test_images.extend(scenes[s])

def get_anns(img_list, all_anns):
    img_ids = set([img['id'] for img in img_list])
    return [ann for ann in all_anns if ann['image_id'] in img_ids]

train_data = {"images": train_images, "annotations": get_anns(train_images, annotations), "categories": categories}
test_data = {"images": test_images, "annotations": get_anns(test_images, annotations), "categories": categories}

with open('seaclear_train.json', 'w') as f:
    json.dump(train_data, f)

with open('seaclear_test.json', 'w') as f:
    json.dump(test_data, f)

print(f"✅ Split සාර්ථකයි!\nTraining Images: {len(train_images)}\nTesting Images: {len(test_images)}")