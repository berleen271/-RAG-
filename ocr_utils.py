import os
import sys
import subprocess
import pytesseract

def find_tesseract():
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path
    if sys.platform.startswith("win"):
        candidates = [
            r"D:\新建文件夹\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
    else:
        candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def setup_ocr():
    path = find_tesseract()
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
        return True
    return False