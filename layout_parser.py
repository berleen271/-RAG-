import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import fitz
import re
import numpy as np
from PIL import Image
import requests
import json
import pytesseract
from config import QWEN_API_KEY, QWEN_URL_VL
from utils import TempImageManager

def detect_heading(text, font_size, is_bold):
    if re.match(r"^(第[一二三四五六七八九十\d]+章|[IVXLCDM]+\.|\d+(\.\d+)*)\s", text):
        return True
    if font_size > 13 and is_bold:
        return True
    return False

def semantic_chunk_page(page):
    text_dict = page.get_text("dict")
    blocks = text_dict["blocks"]
    blocks_sorted = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]) if "bbox" in b else (0,0))
    chunks = []
    current = {"type": "text", "content": "", "bbox": None}
    for block in blocks_sorted:
        if block["type"] == 0:
            for line in block["lines"]:
                line_text = ""
                line_bbox = None
                avg_font = 0
                is_bold = False
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text: continue
                    line_text += text + " "
                    avg_font += span["size"]
                    if "Bold" in span["font"]: is_bold = True
                    if line_bbox is None:
                        line_bbox = list(span["bbox"])
                    else:
                        line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                        line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                        line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                        line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                line_text = line_text.strip()
                if not line_text: continue
                avg_font = avg_font / len(line["spans"]) if line["spans"] else 10
                if detect_heading(line_text, avg_font, is_bold):
                    if current["content"].strip():
                        chunks.append(current)
                    current = {"type": "heading", "content": line_text, "bbox": line_bbox}
                else:
                    if current["bbox"] is None:
                        current["bbox"] = line_bbox
                    else:
                        if line_bbox:
                            current["bbox"][0] = min(current["bbox"][0], line_bbox[0])
                            current["bbox"][1] = min(current["bbox"][1], line_bbox[1])
                            current["bbox"][2] = max(current["bbox"][2], line_bbox[2])
                            current["bbox"][3] = max(current["bbox"][3], line_bbox[3])
                    if current["type"] == "heading":
                        current["content"] += " " + line_text
                        current["type"] = "text"
                    else:
                        current["content"] += " " + line_text
        elif block["type"] == 1:
            if current["content"].strip():
                chunks.append(current)
            current = {"type": "text", "content": "", "bbox": None}
            chunks.append({"type": "figure", "content": "", "bbox": block["bbox"]})
    if current["content"].strip():
        chunks.append(current)
    return chunks

def extract_layout_structured(page, page_num, zoom, img_manager, clip_proc, clip_model, device, tesseract_available):
    elements = []
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    words = page.get_text("words")
    if words:
        xs = [w[0] for w in words] + [w[2] for w in words]
        ys = [w[1] for w in words] + [w[3] for w in words]
        text_rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        coverage = (text_rect.width * text_rect.height) / (page.rect.width * page.rect.height)
    else:
        coverage = 0.0
    ocr_text = ""
    if coverage < 0.1 and tesseract_available:
        try:
            ocr_text = pytesseract.image_to_string(page_img, lang="chi_sim+eng")
        except:
            ocr_text = ""
    chunks = semantic_chunk_page(page)
    if not chunks and ocr_text.strip():
        chunks = [{"type": "text", "content": ocr_text, "bbox": (0,0,page.rect.width*zoom,page.rect.height*zoom)}]
    for chunk in chunks:
        if chunk["type"] in ["heading", "text"] and len(chunk["content"]) > 10:
            elements.append({
                "type": chunk["type"], "page": page_num,
                "content": chunk["content"],
                "bbox": chunk["bbox"] if chunk["bbox"] else (0,0,0,0),
                "img_path": None
            })
    tab_finder = page.find_tables()
    if tab_finder:
        for t in tab_finder.tables:
            data = t.extract()
            if not data or len(data) < 2: continue
            md = "\n".join(["| " + " | ".join([str(c).strip() if c else "" for c in r]) + " |" for r in data])
            tab_bbox = t.bbox
            crop = page_img.crop((tab_bbox[0]*zoom, tab_bbox[1]*zoom, tab_bbox[2]*zoom, tab_bbox[3]*zoom))
            if crop is None or crop.width <= 0 or crop.height <= 0:
                continue
            img_path = img_manager.save(crop, page_num, len(elements))
            elements.append({
                "type": "table", "page": page_num,
                "content": f"[TABLE]\n{md}",
                "bbox": tab_bbox, "img_path": img_path
            })
    for img_info in page.get_images():
        bbox = page.get_image_bbox(img_info)
        if bbox is None: continue
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        area_ratio = (w*h) / (page.rect.width * page.rect.height)
        if area_ratio < 0.05 or bbox[1] < page.rect.height*0.1 or bbox[3] > page.rect.height*0.9:
            continue
        clip_bbox = (bbox[0]-20, bbox[3]+5, bbox[2]+20, bbox[3]+40)
        caption = page.get_text("text", clip=clip_bbox).strip()
        is_figure = bool(re.search(r"(图|Fig|Figure|Chart)\s*\d+", caption, re.I))
        if not (is_figure or area_ratio > 0.1): continue
        crop = page_img.crop((bbox[0]*zoom, bbox[1]*zoom, bbox[2]*zoom, bbox[3]*zoom))
        if crop is None or crop.width <= 0 or crop.height <= 0:
            continue
        chart_types = ["chart", "table", "photo", "logo", "formula"]
        text_inputs = clip_proc(text=chart_types, return_tensors="pt", padding=True).to(device)
        img_inputs = clip_proc(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model(**img_inputs, **text_inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        pred_type = chart_types[np.argmax(probs)]
        content = "[IMAGE]"
        img_path = img_manager.save(crop, page_num, len(elements))
        if pred_type in ["chart", "table"]:
            b64 = pil_to_b64(crop)
            json_text = None
            for attempt in range(2):
                try:
                    resp = requests.post(
                        QWEN_URL_VL,
                        headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "qwen-vl-plus",
                            "input": {
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"image": f"data:image/png;base64,{b64}"},
                                        {"text": "提取图表数据，仅输出纯JSON：{\"x\":[],\"y\":[],\"trend\":\"\"}，不要Markdown。"}
                                    ]
                                }]
                            }
                        },
                        timeout=8
                    )
                    raw = resp.json()["output"]["choices"][0]["message"]["content"]
                    raw = re.sub(r"```json|```", "", raw).strip()
                    parsed = json.loads(raw)
                    if all(k in parsed for k in ["x", "y", "trend"]):
                        json_text = json.dumps(parsed, ensure_ascii=False)
                        break
                except:
                    pass
            if json_text:
                content = f"[CHART_STRUCTURED]\n{json_text}"
            else:
                # 普通图片：生成文字描述
                b64 = pil_to_b64(crop)
                caption_text = None
                try:
                    resp = requests.post(
                        QWEN_URL_VL,
                        headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "qwen-vl-plus",
                            "input": {
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"image": f"data:image/png;base64,{b64}"},
                                        {"text": "用中文简要描述这张图像的内容、对象关系和关键信息，限100字以内。"}
                                    ]
                                }]
                            }
                        },
                        timeout=8
                    )
                    caption_text = resp.json()["output"]["choices"][0]["message"]["content"].strip()
                except:
                    pass
                if caption_text:
                    content = f"[IMAGE_CAPTION]\n{caption_text}"
                else:
                    content = "[IMAGE]（未能生成描述）"
    return elements

# 导入工具函数需在函数内部或顶层，这里从utils导入
from utils import pil_to_b64