import json
from collections import defaultdict


def merge_coco_grounding(captions_file, instances_file, output_file, caption_index=0):
    """
    将 COCO captions 和 instances 合并

    Args:
        captions_file (str): captions_train2017.json 文件路径
        instances_file (str): instances_train2017.json 或 val 文件路径
        output_file (str): 输出文件路径
        caption_index (int): 如果一张图像有多个描述，使用第几个（默认 0，即第一个）
    """
    # 1. 加载 captions，建立 image_id -> list of captions 的映射
    print("正在加载 captions 文件...")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)

    image_captions = defaultdict(list)
    for ann in captions_data['annotations']:
        image_captions[ann['image_id']].append(ann['caption'])
    print(f"已加载 {len(image_captions)} 张图像的描述。")

    # 2. 加载 instances
    print("正在加载 instances 文件...")
    with open(instances_file, 'r', encoding='utf-8') as f:
        instances_data = json.load(f)

    # 3. 构建新的 annotations 列表，为每个标注添加 caption
    new_annotations = []
    skipped = 0
    for ann in instances_data['annotations']:
        img_id = ann['image_id']
        if img_id in image_captions:
            # 选择指定索引的 caption（如果索引超出范围，取最后一个）
            captions = image_captions[img_id]
            if caption_index < len(captions):
                caption = captions[caption_index]
            else:
                caption = captions[-1]
            # 复制原标注并添加 caption 字段
            new_ann = ann.copy()
            new_ann['caption'] = caption
            new_annotations.append(new_ann)
        else:
            skipped += 1

    print(f"已处理 {len(new_annotations)} 个标注，{skipped} 个标注因缺少描述被跳过。")

    # 4. 构建输出数据
    output_data = {
        'images': instances_data['images'],
        'categories': instances_data['categories'],
        'annotations': new_annotations
    }

    # 5. 写入新文件
    print(f"正在写入输出文件 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print("完成！")


if __name__ == "__main__":
    # 使用示例（请根据实际文件路径修改）
    merge_coco_grounding(
        captions_file=r"D:\COCO\annotations_train_val\captions_val2017.json",
        instances_file=r"D:\COCO\annotations_train_val\instances_val2017.json",
        output_file=r"D:\COCO\annotations_train_val\val_labels.json",
        caption_index=0  # 使用第一个描述
    )