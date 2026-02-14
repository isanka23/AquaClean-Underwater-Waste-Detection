import json
import xml.etree.ElementTree as ET
import os

def convert_to_coco(xml_folder, output_json): 
    """
    Converts Pascal VOC XML annotations to a single COCO JSON file.
    Filters out 'timestamp' and other non-relevant metadata labels.
    """
    # Define the classes relevant to your FYP research
    categories = ["plastic", "bio", "rov"]
    cat_to_id = {name: i for i, name in enumerate(categories)}
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
    }
    
    ann_id = 1
    processed_count = 0

    if not os.path.exists(xml_folder):
        print(f"❌ Error: The folder {xml_folder} does not exist.")
        return

    print(f"⏳ Starting conversion from {xml_folder}...")

    # Loop through all XML files in the directory
    for img_id, xml_file in enumerate(os.listdir(xml_folder)):
        if not xml_file.endswith('.xml'): 
            continue
        
        try:
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            
            # Add Image info to COCO structure
            coco_output["images"].append({
                "id": img_id,
                "file_name": root.find('filename').text,
                "width": int(root.find('size/width').text),
                "height": int(root.find('size/height').text)
            })
            
            # Add Annotation info for specific objects
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                
                # This 'if' statement is the filter that removes 'timestamp'
                if cls_name not in cat_to_id: 
                    continue 
                
                bbox = obj.find('bndbox')
                x1, y1 = float(bbox.find('xmin').text), float(bbox.find('ymin').text)
                x2, y2 = float(bbox.find('xmax').text), float(bbox.find('ymax').text)
                
                # COCO format requires: [x_min, y_min, width, height]
                w, h = x2 - x1, y2 - y1
                
                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_to_id[cls_name],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
            processed_count += 1
        except Exception as e:
            print(f"⚠️ Skipping {xml_file} due to error: {e}")

    # Save the final JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_output, f)
    
    print(f"✅ Successfully created: {output_json}")
    print(f"📊 Processed {processed_count} XML files.")
    print(f"📦 Total objects kept (plastic, bio, rov): {ann_id - 1}")

# --- 🚀 RUN CONVERSION ---
# This uses your specific local project paths
xml_input_folder = r"D:\4th year\fyp\AquaClean_Project\trash_ICRA19\trash_ICRA19\dataset\train"
output_filename = "instances_train.json"

if __name__ == "__main__":
    convert_to_coco(xml_input_folder, output_filename)
