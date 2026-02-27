import json
import argparse
from collections import defaultdict

def split_odvg_by_class(input_file, output_file_0_64, output_file_65_79):
    """
    将 ODVG 格式的 .jsonl 文件按类别 ID 范围拆分。
    每个图像可能出现在两个输出文件中（如果同时包含两类标注），
    但每个文件中的实例仅保留对应范围内的标注。
    """
    # 打开输出文件
    f0 = open(output_file_0_64, 'w', encoding='utf-8')
    f65 = open(output_file_65_79, 'w', encoding='utf-8')

    # 计数器
    stats = defaultdict(int)  # total, range0, range65, both

    with open(input_file, 'r', encoding='utf-8') as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析失败，已跳过。错误：{e}")
                continue

            # 确保图像包含检测标注
            if 'detection' not in item or 'instances' not in item['detection']:
                # 如果没有检测标注，可选择跳过或保留原样（这里选择跳过）
                continue

            instances = item['detection']['instances']
            # 分别收集两个范围内的实例
            inst_0_64 = []
            inst_65_79 = []
            for inst in instances:
                label = inst.get('label')
                if label is None:
                    continue
                if 0 <= label <= 64:
                    inst_0_64.append(inst)
                elif 65 <= label <= 79:
                    inst_65_79.append(inst)
                else:
                    # 超出范围的情况（理论上不会发生，但以防万一）
                    print(f"警告：图像 {item.get('filename', 'unknown')} 包含超出 0-79 的标签 {label}，已忽略。")

            # 统计
            stats['total'] += 1
            if inst_0_64:
                stats['range0'] += 1
            if inst_65_79:
                stats['range65'] += 1
            if inst_0_64 and inst_65_79:
                stats['both'] += 1

            # 写入范围0-64的文件
            if inst_0_64:
                # 创建新条目（浅拷贝并替换 instances）
                new_item = item.copy()
                new_item['detection'] = item['detection'].copy()  # 浅拷贝 detection 字典
                new_item['detection']['instances'] = inst_0_64
                f0.write(json.dumps(new_item, ensure_ascii=False) + '\n')

            # 写入范围65-79的文件
            if inst_65_79:
                new_item = item.copy()
                new_item['detection'] = item['detection'].copy()
                new_item['detection']['instances'] = inst_65_79
                f65.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    f0.close()
    f65.close()

    # 输出统计信息
    print("拆分完成。统计信息：")
    print(f"  总图像数：{stats['total']}")
    print(f"  包含类别 0-64 的图像数：{stats['range0']}")
    print(f"  包含类别 65-79 的图像数：{stats['range65']}")
    print(f"  同时包含两类的图像数（重复计数）：{stats['both']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将 ODVG 格式文件按类别 ID 范围拆分为两个文件。")
    parser.add_argument('--input', '-i', required=True, help='输入的 .jsonl 文件路径')
    parser.add_argument('--output-0-64', '-o1', required=True, help='输出文件（类别 0-64）路径')
    parser.add_argument('--output-65-79', '-o2', required=True, help='输出文件（类别 65-79）路径')
    args = parser.parse_args()

    split_odvg_by_class(args.input, args.output_0_64, args.output_65_79)