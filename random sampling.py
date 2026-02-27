import json
import random


def main():
    input_file = r"F:\Open-GroundingDino-main\label\train_labels_odvg_train_0-64.json"

    # 读取所有标注数据
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # 建立图片名到完整数据的映射
    image_map = {item['filename']: item for item in data}
    all_images = list(image_map.keys())

    # 统计每张图片包含的类别（去重）
    image_categories = {}
    for img in all_images:
        item = image_map[img]
        categories = set()
        for inst in item['detection']['instances']:
            categories.add(inst['category'])
        image_categories[img] = list(categories)

    # 统计每个类别对应的图片列表
    cat_images = {}
    for img, cats in image_categories.items():
        for cat in cats:
            cat_images.setdefault(cat, []).append(img)

    all_categories = list(cat_images.keys())
    print(f"总共有 {len(all_categories)} 个类别")

    # 选择划分比例
    print("\n请选择训练集所占比例（训练:总数据）：")
    print("1: 1/3 (即 1:2)")
    print("2: 1/4 (即 1:3)")
    print("3: 1/5 (即 1:4)")
    print("4: 1/6 (即 1:5)")
    choice = input("请输入数字1-4：").strip()
    ratio_map = {
        '1': (1 / 3, '1_2'),
        '2': (1 / 4, '1_3'),
        '3': (1 / 5, '1_4'),
        '4': (1 / 6, '1_5')
    }
    if choice not in ratio_map:
        print("无效选择，默认使用 1/3")
        train_ratio = 1 / 3
        ratio_str = '1_2'
    else:
        train_ratio, ratio_str = ratio_map[choice]

    # 第一步：确保每个类别至少有一张图片被选中
    must_have = set()
    for cat, imgs in cat_images.items():
        # 随机选择该类别的一张图片
        selected = random.choice(imgs)
        must_have.add(selected)

    # 剩余图片随机打乱后，再按比例补足
    remaining = [img for img in all_images if img not in must_have]
    random.shuffle(remaining)
    total_needed = int(len(all_images) * train_ratio)
    additional_needed = max(0, total_needed - len(must_have))
    selected_images = must_have | set(remaining[:additional_needed])

    # 最终训练集
    train_set = selected_images

    # 检查每个类别是否都在训练集中
    train_cats = set()
    for img in train_set:
        train_cats.update(image_categories[img])
    missing_cats = set(all_categories) - train_cats
    if missing_cats:
        print(f"警告：以下类别在训练集中缺失：{missing_cats}")
    else:
        print("所有类别均已覆盖")

    # 写入训练集文件
    train_file = rf"F:\Open-GroundingDino-main\label\train_labels_odvg_train_0-64_{ratio_str}.json"
    with open(train_file, 'w') as f:
        for img in sorted(train_set):
            json.dump(image_map[img], f)
            f.write('\n')

    print(f"\n划分完成！训练集共 {len(train_set)} 张图片（占总图片数的 {len(train_set) / len(all_images):.2%}）")
    print(f"输出文件：{train_file}")


if __name__ == "__main__":
    main()