from groundingdino.util.inference import load_model, load_image, predict
import os
from tqdm import tqdm
import json
import numpy as np
import torch
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载模型
model = load_model(
    r"D:\GroundingDINO-main\GroundingDINO-main\groundingdino\config\GroundingDINO_SwinB_cfg.py",
    r"D:\GroundingDINO-main\GroundingDINO-main\weights\groundingdino_swinb_cogcoor.pth"
)

# 构造提示词
remote = ["a device used to control TV or other electronics", "a remote control with buttons", "a handheld remote for television", "an infrared remote controller", "a black plastic remote"]
keyboard = ["a computer keyboard with keys", "a typing device for computers", "a QWERTY keyboard", "a peripheral with keys for input", "a wireless keyboard"]
cell_phone = ["a mobile phone with a screen", "a smartphone device", "a handheld cellular phone", "a mobile communication device", "a flip phone"]
micro_wave = ["a microwave oven for heating food", "a kitchen appliance for reheating meals", "a microwave with a digital display", "a countertop microwave", "an appliance used to cook food quickly"]
oven = ["a kitchen oven for baking", "a gas oven with a door", "an electric oven for cooking", "a built-in oven", "a stove oven combination"]
toaster = ["a toaster for browning bread", "an electric toaster with slots", "a kitchen appliance for toasting", "a pop-up toaster", "a toaster for making toast"]
sink = ["a kitchen sink with faucet", "a kitchen sink with faucet", "a stainless steel sink", "a basin for washing dishes", "a sink with a drain"]
refrigerator = ["a refrigerator for storing food", "a fridge with freezer compartment", "a kitchen refrigerator", "a stainless steel fridge", "an appliance that keeps food cold"]
book = ["a book with pages", "a hardcover book", "a novel for reading", "a stack of books", "an open book"]
clock = ["a wall clock with numbers", "an analog clock with hands", "a digital clock displaying time", "a bedside alarm clock", "a grandfather clock"]
vase = ["a vase for flowers", "a vase for flowers", "a glass vase", "a decorative flower vase", "a tall vase"]
scissors = ["a pair of scissors", "scissors for cutting paper", "a cutting tool with two blades", "a metal scissors", "a pair of shears"]
teddy_bear = ["a stuffed teddy bear", "a soft plush bear toy", "a brown teddy bear", "a cuddly bear for children", "a toy bear"]
hair_drier = ["a hair dryer for drying hair", "an electric hair drier", "a handheld hair dryer", "a blow dryer", "a hair styling tool"]
toothbrush = ["a toothbrush for cleaning teeth", "an electric toothbrush", "a manual toothbrush", "a toothbrush with bristles", "a dental hygiene tool"]
prompt = [remote, keyboard, cell_phone, micro_wave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_drier, toothbrush]

# 类别名称列表（必须与prompt顺序一致）
class_names = [
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 推理时会用到的一些参数
box_threshold = 0.25
text_threshold = 0.25
iou_threshold = 0.5  # NMS的IoU阈值

# 获取图像信息
os.chdir(r"D:\COCO\val_img\val2017")
imgs = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 读取标签文件,并获取图像信息
with open(r"D:\COCO\annotations_train_val\val_labels_unseen_cocoformat(15).json", 'r') as f:
    info = json.load(f)
images_info = {img['id']: (img['width'], img['height']) for img in info['images']}
categories_info = {cate['name']: cate['id'] for cate in info['categories']}

# 输出结果文件
output_json = r"D:\Zero-Shot Object Detection\ZSD-Prompt Ensembling\fusion_results_nms.json"
all_results = []  # 存储所有图片的最终检测结果

# 辅助函数：坐标转换
def cxcywh_to_xyxy(boxes_norm, img_w, img_h):
    """归一化[cx,cy,w,h] -> 绝对坐标[x1,y1,x2,y2]"""
    boxes = boxes_norm * np.array([img_w, img_h, img_w, img_h])
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def xyxy_to_xywh(boxes_xyxy):
    """[x1,y1,x2,y2] -> [x,y,w,h]（左上角坐标）"""
    x = boxes_xyxy[:, 0]
    y = boxes_xyxy[:, 1]
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    return np.stack([x, y, w, h], axis=1)

# 遍历所有图像并做预测
for img in tqdm(imgs):
    # 获取文件id
    file_name = os.path.splitext(os.path.basename(img))[0]
    image_id = int(file_name)
    if image_id not in images_info:
        continue

    img_w, img_h = images_info[image_id]
    # 用于存储当前图片中每个类别的检测框（按类别分别收集）
    per_cat_candidates = {}

    for class_idx, prompt_type in enumerate(prompt):
        class_name = class_names[class_idx]
        cat_id = categories_info[class_name]

        # 初始化该类别的候选列表
        if cat_id not in per_cat_candidates:
            per_cat_candidates[cat_id] = {"boxes_xyxy": [], "scores": []}

        # 对该类别的每个提示词进行推理
        for single_prompt in prompt_type:
            # 加载图片（效率较低，但保留原始风格）
            image_source, image_tensor = load_image(img)
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=single_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if len(boxes) == 0:
                continue
            # 转换为绝对坐标 [x1,y1,x2,y2]
            boxes_xyxy = cxcywh_to_xyxy(boxes.numpy(), img_w, img_h)
            scores = logits.numpy()
            # 添加到该类别的候选列表
            per_cat_candidates[cat_id]["boxes_xyxy"].append(boxes_xyxy)
            per_cat_candidates[cat_id]["scores"].append(scores)

    # 对每个类别进行NMS融合
    for cat_id, cand in per_cat_candidates.items():
        if len(cand["boxes_xyxy"]) == 0:
            continue
        # 合并该类别的所有框
        boxes_xyxy = np.vstack(cand["boxes_xyxy"])
        scores = np.hstack(cand["scores"])
        # NMS
        keep = nms(
            torch.from_numpy(boxes_xyxy).float(),
            torch.from_numpy(scores).float(),
            iou_threshold
        ).numpy()
        # 转换为COCO格式的xywh
        boxes_xywh = xyxy_to_xywh(boxes_xyxy[keep])
        scores_keep = scores[keep]
        for box, score in zip(boxes_xywh, scores_keep):
            result_item = {
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": box.tolist(),
                "score": float(score)
            }
            all_results.append(result_item)

# 保存结果
with open(output_json, 'w') as f:
    json.dump(all_results, f, indent=4)
print(f"预测结果已保存至：{output_json}")

# 计算mAP
coco_gt = COCO(r"D:\COCO\annotations_train_val\val_labels_unseen_cocoformat(15).json")
coco_dt = coco_gt.loadRes(output_json)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
print("\n" + "="*50)
print("NMS融合后的评估结果：")
coco_eval.summarize()