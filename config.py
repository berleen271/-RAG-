import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-YOUR-KEY-HERE")
QWEN_URL_TEXT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_URL_VL   = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
CHROMA_PATH = "./chroma_v12"

