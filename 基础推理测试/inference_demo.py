from groundingdino.util.inference import load_model, load_image, predict , annotate
import cv2
from pathlib import Path
import os

parent_dir = Path(__file__).resolve().parent.parent
os.chdir(parent_dir)
# 1. 加载模型(使用预训练模型)
model = load_model(
    r"./GroundingDINO-main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
r"./GroundingDINO-main/weights/groundingdino_swinb_cogcoor.pth"
)


# 2. 读取图片
image_source, image = load_image(r"./基础推理测试/demo pic.png")

# 3. 文本提示
TEXT_PROMPT = "a dog.a cat."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# 4. 推理
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)
result = {'bbox':boxes, 'logits':logits, 'phrases':phrases}
print(result)

# 标注图像
annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

# 保存结果
cv2.imwrite("./基础推理测试/detection_result.jpg", annotated_frame)
print("检测结果已保存到 detection_result.jpg")
