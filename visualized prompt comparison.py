from groundingdino.util.inference import load_model, load_image, predict , annotate
import cv2

model = load_model(
    r"./GroundingDINO-main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
r"./GroundingDINO-main/weights/groundingdino_swinb_cogcoor.pth"
)

image_source_remote, image_remote = load_image(r"D:\COCO\val_img\val2017\000000476810.jpg")
image_source_keyboard, image_keyboard = load_image(r"D:\COCO\val_img\val2017\000000108026.jpg")

prompt_remote=["remote.","a photo of remote.","a handheld electronic remote control with multiple buttons."]
prompt_keyboard=["keyboard.","a photo of keyboard.","a computer keyboard with rectangular keys arranged in rows used for typing on a desktop or laptop computer."]

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

category=["pure class name","template sentence","fine granularity description"]

for i in range(3):
    boxes_remote, logits_remote, phrases_remote = predict(
        model=model,
        image=image_remote,
        text_threshold=TEXT_THRESHOLD,
        box_threshold=BOX_THRESHOLD,
        caption=prompt_remote[i],
    )

    boxes_keyboard, logits_keyboard, phrases_keyboard = predict(
        model=model,
        image=image_keyboard,
        text_threshold=TEXT_THRESHOLD,
        box_threshold=BOX_THRESHOLD,
         caption=prompt_keyboard[i],
    )

    remote_annotated_frame = annotate(
        image_source=image_source_remote,
        boxes=boxes_remote,
        logits=logits_remote,
        phrases=phrases_remote
    )

    keyboard_annotated_frame = annotate(
        image_source=image_source_keyboard,
        boxes=boxes_keyboard,
        logits=logits_keyboard,
        phrases=phrases_keyboard
    )
    cv2.imwrite(f"./visualized prompts comparison results/remote_detection_{category[i]}.jpg", remote_annotated_frame)
    cv2.imwrite(f"./visualized prompts comparison results/keyboard_detection_{category[i]}.jpg", keyboard_annotated_frame)