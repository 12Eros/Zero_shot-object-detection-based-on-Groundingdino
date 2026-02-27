from groundingdino.util.inference import load_model,load_image,predict,annotate
import os
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载模型
model = load_model(
    "./GroundingDINO-main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
r"./GroundingDINO-main/weights/groundingdino_swinb_cogcoor.pth"
)

# 推理时会用到的一些参数
box_threshold = 0.25
text_threshold = 0.25
prompt = "person . bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush."

# 获取图像信息
os.chdir(r"D:\COCO\val_img\val2017")
imgs = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 读取标签文件,并获取图像信息
with open(r"D:\COCO\annotations_train_val\val_labels.json",'r') as f:
    info = json.load(f)
images_info = {img['id']: (img['width'], img['height']) for img in info['images']}
categories_info = {cate['name']:cate['id'] for cate in info['categories']}


results=[]

# 遍历所有图像并做预测，每一次预测后都将预测信息写入json文件
for img in tqdm(imgs):
    image_source, image = load_image(img)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    # 接下来获取文件id
    file_name = os.path.splitext(os.path.basename(img))[0]
    # 去掉前面的0
    image_id = int(file_name)

    # 找到images的高宽
    image_width, image_height = images_info[image_id]
    # 真实坐标
    boxes[:,0] = boxes[:,0]*image_width
    boxes[:,1] = boxes[:,1]*image_height
    boxes[:,2] = boxes[:,2]*image_width
    boxes[:,3] = boxes[:,3]*image_height
    # 转换为[x(左上角),y(左上角),w,h]的形式
    x = boxes[:,0]-boxes[:,2]/2
    y = boxes[:,1]-boxes[:,3]/2
    w = boxes[:,2]
    h = boxes[:,3]

    # 得到bbox的个数
    box_num = len(x)

    # 获取类别id
    categories_id = []
    invalid_categories = []
    for i in range(box_num):
        category_id = categories_info.get(phrases[i])
        categories_id.append(category_id)
        if category_id is None:
            invalid_categories.append(i)
            continue
    # 现在bbox,图片id,类别id和置信度已经处理完毕,开始写入文件中
    #TODO:1.打包成字典
    for k in range(box_num):
        if k in invalid_categories:
            continue
        else:
            inference_result = {
                                   "image_id": image_id,
                                   "category_id": categories_id[k],
                                   "bbox": [float(x[k]),float(y[k]),float(w[k]),float(h[k])],
                                   "score":float(logits[k])
            }
            results.append(inference_result)
#TODO:2.写入json文件
with open(r"./groundingdino_eval_on_COCO/eval_on_coco_results.json",'w') as f:
    json.dump(results,f,indent=4)
#TODO:3.计算mAP值

# 加载 ground truth 标注文件
gt_ann_file = r"D:\COCO\annotations_train_val\val_labels.json"
coco_gt = COCO(gt_ann_file)

# 加载预测结果文件（假设已按上述格式生成）
with open("./groundingdino_eval_on_COCO/eval_on_coco_results.json", 'r') as f:
    results = json.load(f)

# 将预测结果转换为 COCO 可用的对象
coco_dt = coco_gt.loadRes(results)

# 初始化评估器
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')   # 'bbox' 表示检测任务

# 运行评估
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()   # 打印 mAP 等指标











