# register_dataset.py

from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(CURRENT_DIR, "..", "datasets", "labels_bbox")
IMAGE_DIR = os.path.join(CURRENT_DIR, "..", "datasets", "images")

LABEL_DIR = os.path.normpath(LABEL_DIR)
IMAGE_DIR = os.path.normpath(IMAGE_DIR)

# label 이름을 ID에 매핑
def generate_category_name_to_id(label_dir):
    category_set = set()
    
    for file_name in os.listdir(label_dir):
        if not file_name.endswith(".json"):
            continue
        
        file_path = os.path.join(label_dir, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)
            
        for ann in data["annotations"]:
            category_set.add(ann["category"])
            
    category_list = sorted(list(category_set))
    return {name: idx for idx, name in enumerate(category_list)}

# dataset 로더 함수
def load_custom_dataset(json_dir, image_root, category_name_to_id):
    dataset_dicts = []
    print(f"Registering dataset from {json_dir}, image_root = {image_root}")
    
    for idx, file in enumerate(os.listdir(json_dir)):
        if not file.endswith(".json"):
            continue
        print(f"Processing file: {file}")
        with open(os.path.join(json_dir, file), "r") as f:
            data = json.load(f)

        record = {
            "file_name": os.path.join(image_root, data["file_name"]),
            "image_id": idx,
            "height": data["height"],
            "width": data["width"],
            "annotations": []
        }

        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            record["annotations"].append({
                "bbox": [x, y, w, h],
                "bbox_mode": 0,  # BoxMode.XYWH_ABS
                "category_id": category_name_to_id[ann["category"]],
            })

        dataset_dicts.append(record)
    
    print(f"Total loaded records: {len(dataset_dicts)}")
    return dataset_dicts

# 등록
def register_finetuning_dataset():
    category_name_to_id = generate_category_name_to_id(LABEL_DIR)

    DatasetCatalog.register("FineTuning_train",
                            lambda: load_custom_dataset(LABEL_DIR, IMAGE_DIR, category_name_to_id)
    )
    MetadataCatalog.get("FineTuning_train").set(
        thing_classes=list(category_name_to_id.keys()),
        evaluator_type="cityscapes"
    )
