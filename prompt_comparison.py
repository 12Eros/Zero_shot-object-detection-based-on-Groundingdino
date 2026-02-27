from groundingdino.util.inference import load_model,load_image,predict,annotate
import os
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载模型
model = load_model(
    r"./GroundingDINO-main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
r"./GroundingDINO-main/weights/groundingdino_swinb_cogcoor.pth"
)
# 构造提示词
fine_granularity_description = ["a handheld electronic remote control with multiple buttons used to operate a television or home appliance in a living room environment.","a computer keyboard with rectangular keys arranged in rows used for typing on a desktop or laptop computer.","a modern smartphone with a touchscreen display used for communication, messaging, and browsing the internet.","a kitchen microwave oven with a glass door and control panel used for heating or cooking food.","a household oven with a front-opening door and metal racks used for baking or roasting food.","a small electric toaster with two slots used for toasting slices of bread on a kitchen counter.","a kitchen sink with a metal faucet used for washing dishes and food preparation.","a large refrigerator with double doors used for storing food and beverages in a kitchen.","a printed book with pages and a cover used for reading and studying.","a wall-mounted or desk clock with numbers and hands used to display time.","a decorative flower vase made of glass or ceramic used for holding flowers indoors.","a pair of metal scissors with two sharp blades and handles used for cutting paper or fabric.","a soft teddy bear toy made of plush fabric often given to children as a gift.","an electric hair drier with a handle and nozzle used for drying hair after washing.","a toothbrush with a plastic handle and bristles used for cleaning teeth."]
prompt_template_sentence=["a photo of remote","a photo of keyboard.","a photo of cell phone.","a photo of microwave.","a photo of oven.","a photo of toaster.","a photo of sink.","a photo of refrigerator.","a photo of book.","a photo of clock.","a photo of vase.","a photo of scissors.","a photo of teddy bear.","a photo of hair drier.","a photo of toothbrush."]
prompt_pure_class_name = ["remote.","keyboard.","cell phone.","microwave.","oven.","toaster.","sink.","refrigerator.","book.","clock.","vase.","scissors.","teddy bear.","hair drier.","toothbrush."]
prompt_list=[fine_granularity_description,prompt_template_sentence,prompt_pure_class_name]
# 推理时会用到的一些参数
box_threshold = 0.25
text_threshold = 0.25

# 获取图像信息
os.chdir(r"D:\COCO\val_img\val2017")
imgs = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 读取标签文件,并获取图像信息
with open(r"D:\COCO\annotations_train_val\val_labels_unseen_cocoformat(15).json",'r') as f:
    info = json.load(f)
images_info = {img['id']: (img['width'], img['height']) for img in info['images']}
categories_info = {cate['name']:cate['id'] for cate in info['categories']}
# 计数
count = 0
# 遍历所有图像并做预测，每一次预测后都将预测信息写入json文件
for prompt_type in prompt_list:
    results = []
    count += 1
    for prompt in prompt_type:
        for img in tqdm(imgs):
            # 接下来,先获取文件id
            file_name = os.path.splitext(os.path.basename(img))[0]
            # 去掉前面的0
            image_id = int(file_name)
            # 由于数据集的图片不便于划分,对于不含unseen类的图片,依据id剔除
            if image_id not in images_info:
                continue
            image_source, image = load_image(img)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # 找到images的高宽
            image_width, image_height = images_info[image_id]
            # 真实坐标
            boxes[:, 0] = boxes[:, 0] * image_width
            boxes[:, 1] = boxes[:, 1] * image_height
            boxes[:, 2] = boxes[:, 2] * image_width
            boxes[:, 3] = boxes[:, 3] * image_height
            # 转换为[x,y,w,h]的形式
            x = boxes[:, 0] - boxes[:, 2] / 2
            y = boxes[:, 1] - boxes[:, 3] / 2
            w = boxes[:, 2]
            h = boxes[:, 3]

            # 得到bbox的个数
            box_num = len(x)

            # 获取类别id
            for i in range(box_num):
                category_id = None
                phrase = phrases[i].lower().strip()

                for key in sorted(categories_info.keys(), key=len, reverse=True):
                    if key in phrase:
                        category_id = categories_info[key]
                        # 现在bbox,图片id,类别id和置信度已经处理完毕,开始写入文件中
                        # TODO:1.打包成字典
                        inference_result = {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [float(x[i]), float(y[i]), float(w[i]), float(h[i])],
                            "score": float(logits[i])
                        }
                        results.append(inference_result)
                        break
    # TODO:2.写入json文件
    with open(f"./prompt_comparison_results/prompt_category{count}_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    # TODO:3.计算mAP值

    # 加载 ground truth 标注文件
    gt_ann_file = r"D:\COCO\annotations_train_val\val_labels_unseen_cocoformat(15).json"
    coco_gt = COCO(gt_ann_file)

    # 加载预测结果文件（假设已按上述格式生成）
    with open(f"./prompt_comparison_results/prompt_category{count}_results.json", 'r') as f:
        results = json.load(f)

    # 将预测结果转换为 COCO 可用的对象
    coco_dt = coco_gt.loadRes(results)

    # 初始化评估器
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')  # 'bbox' 表示检测任务

    # 运行评估
    print("=" * 25 + f"{count}" + "=" * 50)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # 打印 mAP 等指标


