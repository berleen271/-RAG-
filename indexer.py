import numpy as np
from config import CHROMA_PATH
from chromadb import PersistentClient
from utils import TempImageManager

def get_chroma():
    return PersistentClient(CHROMA_PATH)

def build_page_vectors(page_items, embedder, img_manager):
    st_vecs, clip_vecs = [], []
    for item in page_items:
        if item["type"] in ["heading", "text"] and item["content"]:
            st_vecs.append(embedder.encode_text_st([item["content"]])[0])
            clip_vecs.append(embedder.encode_text_clip([item["content"]])[0])
        elif item["type"] in ["table", "figure"] and item.get("img_path"):
            img = img_manager.load(item["img_path"])
            if img:
                clip_vecs.append(embedder.encode_image_clip([img])[0])
            if not item["content"].startswith("[TABLE]") and not item["content"].startswith("[CHART"):
                st_vecs.append(embedder.encode_text_st([item["content"]])[0])
    p_st = np.mean(st_vecs, axis=0).tolist() if st_vecs else [0.0]*embedder.st_dim
    p_clip = np.mean(clip_vecs, axis=0).tolist() if clip_vecs else [0.0]*embedder.clip_dim
    return p_st, p_clip

def build_index(all_data, embedder, img_manager):
    client = get_chroma()
    for name in ["page_text_idx", "page_visual_idx", "text_idx", "visual_idx"]:
        try: client.delete_collection(name)
        except: pass
    p_tc = client.get_or_create_collection("page_text_idx", embedding_function=None)
    p_vc = client.get_or_create_collection("page_visual_idx", embedding_function=None)
    t_c  = client.get_or_create_collection("text_idx", embedding_function=None)
    v_c  = client.get_or_create_collection("visual_idx", embedding_function=None)

    p_t_ids, p_t_vecs, p_t_meta = [], [], []
    p_v_ids, p_v_vecs, p_v_meta = [], [], []
    t_ids, t_vecs, t_docs, t_meta = [], [], [], []
    v_ids, v_vecs, v_docs, v_meta = [], [], [], []

    for pnum, items in all_data.items():
        p_st, p_clip = build_page_vectors(items, embedder, img_manager)
        p_t_ids.append(f"p_{pnum}"); p_t_vecs.append(p_st); p_t_meta.append({"page":pnum})
        p_v_ids.append(f"p_{pnum}"); p_v_vecs.append(p_clip); p_v_meta.append({"page":pnum})

        for idx, it in enumerate(items):
            meta = {
                "page": pnum, "type": it["type"],
                "bbox": str(it["bbox"]),
                "img_path": it.get("img_path") or ""
            }
            if it["content"]:
                vec = embedder.encode_text_st([it["content"]])[0]
                t_ids.append(f"t_{len(t_ids)}"); t_vecs.append(vec)
                t_docs.append(it["content"]); t_meta.append(meta)
            if it["type"] in ["table", "figure"] and it.get("img_path"):
                img = img_manager.load(it["img_path"])
                if img:
                    v_vec = embedder.encode_image_clip([img])[0]
                    v_ids.append(f"v_{len(v_ids)}"); v_vecs.append(v_vec)
                    v_docs.append(it["content"]); v_meta.append(meta)

    if p_t_ids: p_tc.add(ids=p_t_ids, embeddings=p_t_vecs, metadatas=p_t_meta)
    if p_v_ids: p_vc.add(ids=p_v_ids, embeddings=p_v_vecs, metadatas=p_v_meta)
    if t_ids: t_c.add(ids=t_ids, embeddings=t_vecs, documents=t_docs, metadatas=t_meta)
    if v_ids: v_c.add(ids=v_ids, embeddings=v_vecs, documents=v_docs, metadatas=v_meta)
    return p_tc, p_vc, t_c, v_c