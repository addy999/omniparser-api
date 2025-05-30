from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import io
from PIL import Image
import torch
import numpy as np
import os
import uuid

# Existing imports
import numpy as np
import torch
from PIL import Image
import io

from utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)
import torch

# yolo_model = get_yolo_model(model_path='/data/icon_detect/best.pt')
# caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="/data/icon_caption_florence")

from ultralytics import YOLO

# if not os.path.exists("/data/icon_detect"):
#     os.makedirs("/data/icon_detect")

try:
    yolo_model = YOLO("weights/icon_detect/best.pt").to("cuda")
except:
    yolo_model = YOLO("weights/icon_detect/best.pt")

from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
except:
    model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
caption_model_processor = {"processor": processor, "model": model}
print("finish loading model!!!")

app = FastAPI()


class ProcessResponse(BaseModel):
    image: str  # Base64 encoded image
    parsed_content_list: str
    label_coordinates: str

def generate_uuid():
    return str(uuid.uuid4())

def process(
    image_input: Image.Image, box_threshold: float, iou_threshold: float
) -> ProcessResponse:
    image_save_path = f"imgs/{generate_uuid()}.png"
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=True,
    )
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
    )
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print("finish processing")
    parsed_content_list_str = "\n".join(parsed_content_list)

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    os.remove(image_save_path)

    return ProcessResponse(
        image=img_str,
        parsed_content_list=str(parsed_content_list_str),
        label_coordinates=str(label_coordinates),
    )


@app.post("/process_image", response_model=ProcessResponse)
async def process_image(
    image_file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
):
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    response = process(image_input, box_threshold, iou_threshold)
    return response
