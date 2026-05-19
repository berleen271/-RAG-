import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import tempfile
import shutil
import io
import base64
import hashlib
from PIL import Image

class TempImageManager:
    def __init__(self):
        self.dir = None

    def init_dir(self):
        if self.dir and os.path.exists(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = tempfile.mkdtemp(prefix="rag_imgs_")

    def save(self, pil_img, page_num, element_idx):
        if self.dir is None:
            self.init_dir()
        filename = f"p{page_num}_{element_idx}.png"
        path = os.path.join(self.dir, filename)
        pil_img.save(path)
        return filename

    def load(self, img_path):
        if not img_path or self.dir is None:
            return None
        full = os.path.join(self.dir, img_path)
        if os.path.exists(full):
            return Image.open(full)
        return None

    def cleanup(self):
        if self.dir and os.path.exists(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def get_file_hash(b):
    return hashlib.md5(b).hexdigest()