import json
from collections import defaultdict

# ==========================
INPUT_JSON = r"D:\COCO\annotations_train_val\val_labels.json"
OUTPUT_SEEN_JSON = r"D:\COCO\annotations_train_val\val_labels_seen_cocoformat(65).json"
OUTPUT_UNSEEN_JSON = r"D:\COCO\annotations_train_val\val_labels_unseen_cocoformat(15).json"
# ==========================

SEEN_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74
]

# =================================


def split_coco_manual():
    coco = json.load(open(INPUT_JSON, "r", encoding="utf-8"))

    categories = coco["categories"]
    annotations = coco["annotations"]
    images = coco["images"]

    all_category_ids = {c["id"] for c in categories}
    seen_ids = set(SEEN_CATEGORY_IDS)
    unseen_ids = all_category_ids - seen_ids

    print("Seen:", len(seen_ids))
    print("Unseen:", len(unseen_ids))

    # 构建旧id -> 新id 映射（连续）
    seen_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(seen_ids))}
    unseen_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unseen_ids))}

    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    seen_images = []
    unseen_images = []
    seen_annotations = []
    unseen_annotations = []

    for img in images:
        anns = img_to_anns[img["id"]]

        has_seen = any(ann["category_id"] in seen_ids for ann in anns)
        has_unseen = any(ann["category_id"] in unseen_ids for ann in anns)

        if has_seen:
            seen_images.append(img)
            for ann in anns:
                if ann["category_id"] in seen_ids:
                    new_ann = ann.copy()
                    new_ann["category_id"] = seen_id_map[ann["category_id"]]
                    seen_annotations.append(new_ann)

        if has_unseen:
            unseen_images.append(img)
            for ann in anns:
                if ann["category_id"] in unseen_ids:
                    new_ann = ann.copy()
                    new_ann["category_id"] = unseen_id_map[ann["category_id"]]
                    unseen_annotations.append(new_ann)

    # 重建 categories（连续id）
    seen_categories = []
    for c in categories:
        if c["id"] in seen_ids:
            new_c = c.copy()
            new_c["id"] = seen_id_map[c["id"]]
            seen_categories.append(new_c)

    unseen_categories = []
    for c in categories:
        if c["id"] in unseen_ids:
            new_c = c.copy()
            new_c["id"] = unseen_id_map[c["id"]]
            unseen_categories.append(new_c)

    seen_data = {
        "images": seen_images,
        "annotations": seen_annotations,
        "categories": seen_categories
    }

    unseen_data = {
        "images": unseen_images,
        "annotations": unseen_annotations,
        "categories": unseen_categories
    }

    json.dump(seen_data, open(OUTPUT_SEEN_JSON, "w", encoding="utf-8"), indent=2)
    json.dump(unseen_data, open(OUTPUT_UNSEEN_JSON, "w", encoding="utf-8"), indent=2)

    print("划分完成")
    print("Seen类别连续编号: 0 -", len(seen_categories)-1)
    print("Unseen类别连续编号: 0 -", len(unseen_categories)-1)


if __name__ == "__main__":
    split_coco_manual()
