import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def retrieve_hierarchical(query, p_tc, p_vc, t_c, v_c, embedder, reranker,
                          use_page=True, use_vis=True, dynamic_weight=True, top_k=20,
                          return_trace=False):
    q_st = embedder.encode_text_st([query])[0]
    q_clip = embedder.encode_text_clip([query])[0]

    cand_pages = set()
    if use_page:
        p_t_res = p_tc.query(query_embeddings=[q_st], n_results=3)
        cand_pages.update([m["page"] for m in p_t_res["metadatas"][0]])
        if use_vis:
            p_v_res = p_vc.query(query_embeddings=[q_clip], n_results=3)
            cand_pages.update([m["page"] for m in p_v_res["metadatas"][0]])
    else:
        t_all = t_c.query(query_embeddings=[q_st], n_results=top_k*2)
        v_all = v_c.query(query_embeddings=[q_clip], n_results=top_k)
        cand_pages = set([m["page"] for m in t_all["metadatas"][0]] +
                         [m["page"] for m in v_all["metadatas"][0]])
    if not cand_pages:
        if return_trace:
            return [], [], [], [], [], [], [], []
        return []

    t_res = t_c.query(query_embeddings=[q_st], n_results=top_k, where={"page":{"$in":list(cand_pages)}})
    v_res = v_c.query(query_embeddings=[q_clip], n_results=top_k//2, where={"page":{"$in":list(cand_pages)}})

    w_txt, w_vis = 0.6, 0.4
    if dynamic_weight:
        ql = query.lower()
        if any(k in ql for k in ["表","table","数据","金额"]):
            w_txt, w_vis = 0.3, 0.7
        elif any(k in ql for k in ["图","chart","趋势","走势"]):
            w_txt, w_vis = 0.2, 0.8

    def dist2sim(dists):
        return [1.0/(1.0+d) for d in dists]

    txt_sims = dist2sim(t_res["distances"][0]) if t_res["ids"] else []
    vis_sims = dist2sim(v_res["distances"][0]) if v_res["ids"] else []

    merged = []
    for i, m in enumerate(t_res["metadatas"][0]):
        merged.append({
            "score": w_txt * txt_sims[i],
            "src": "text", "page": m["page"],
            "content": t_res["documents"][0][i],
            "bbox_str": m["bbox"], "img_path": m.get("img_path")
        })
    if use_vis:
        for i, m in enumerate(v_res["metadatas"][0]):
            merged.append({
                "score": w_vis * vis_sims[i],
                "src": "visual", "page": m["page"],
                "content": v_res["documents"][0][i],
                "bbox_str": m["bbox"], "img_path": m.get("img_path")
            })
    merged.sort(key=lambda x: x["score"], reverse=True)
    pre_rerank_dicts = [dict(item) for item in merged[:top_k]]

    if len(pre_rerank_dicts) > 1:
        pairs = [[query, c["content"][:400]] for c in pre_rerank_dicts[:15]]
        ce_scores = reranker.predict(pairs)
        for i, c in enumerate(pre_rerank_dicts[:15]):
            c["score"] = float(ce_scores[i])
        pre_rerank_dicts.sort(key=lambda x: x["score"], reverse=True)

    final_context = pre_rerank_dicts[:6]

    if return_trace:
        # 收集 pre-rerank 的得分和内容（merged 已排序）
        pre_scores = [c["score"] for c in merged[:top_k]]
        pre_contents = [c["content"] for c in merged[:top_k]]
        # 收集 post-rerank 的得分和内容（pre_rerank_dicts 前6个）
        post_scores = [c["score"] for c in pre_rerank_dicts[:6]]
        post_contents = [c["content"] for c in pre_rerank_dicts[:6]]
        return (final_context, list(cand_pages), pre_contents, pre_scores,
                post_contents, post_scores, merged[:top_k], final_context)
    return final_context