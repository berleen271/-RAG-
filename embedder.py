import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

def load_models(device_str=None):
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512, device=device)
    return st_model, clip_model, clip_proc, reranker, device

class UnifiedEmbedder:
    def __init__(self, st_model, clip_model, clip_proc, device):
        self.st_model = st_model
        self.clip_model = clip_model
        self.clip_proc = clip_proc
        self.device = device
        self.st_dim = 384
        self.clip_dim = 768

    def encode_text_st(self, texts):
        if isinstance(texts, str): texts = [texts]
        return self.st_model.encode(texts).tolist()

    def encode_text_clip(self, texts):
        if isinstance(texts, str): texts = [texts]
        inputs = self.clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
            feats = outputs.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    def encode_image_clip(self, pil_images):
        if not pil_images: return []
        inputs = self.clip_proc(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            feats = outputs.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()