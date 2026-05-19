import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import pandas as pd
import re
from retriever import retrieve_hierarchical
from generator import generate_answer, verify_and_clean
from evaluation import (evaluate_retrieval_layer, evaluate_chunk_layer,
                        evaluate_answer, evaluate_evidence_layer)
from fact_checker import batch_fact_check

def run_ablation_study(qa_set, p_tc, p_vc, t_c, v_c, embedder, reranker):
    setups = [
        {"name": "Baseline (Flat Text)",      "use_page": False, "use_vis": False, "dynamic": False},
        {"name": "ExpA (+Hierarchical)",      "use_page": True,  "use_vis": False, "dynamic": False},
        {"name": "ExpB (+Visual)",            "use_page": True,  "use_vis": True,  "dynamic": False},
        {"name": "ExpC (+Dynamic Weight)",    "use_page": True,  "use_vis": True,  "dynamic": True},
    ]
    results = []
    for s in setups:
        page_recalls, page_precisions, page_mrrs = [], [], []
        chunk_recalls_pre, chunk_precisions_pre, chunk_mrrs_pre = [], [], []
        chunk_recalls_post, chunk_precisions_post, chunk_mrrs_post = [], [], []
        answer_em, answer_f1, bleu_scores, rouge_scores = [], [], [], []
        ssr_scores = []

        for q, true_ans, true_pages, true_chunks in qa_set:
            final_ctx, pages, pre_dicts, post_dicts = retrieve_hierarchical(
                q, p_tc, p_vc, t_c, v_c, embedder, reranker,
                use_page=s["use_page"], use_vis=s["use_vis"],
                dynamic_weight=s["dynamic"], return_trace=True
            )
            pr = evaluate_retrieval_layer(true_pages, pages, k=3)
            page_recalls.append(pr["Recall@k"]); page_precisions.append(pr["Precision@k"]); page_mrrs.append(pr["MRR"])

            pre_contents = [c["content"] for c in pre_dicts] if pre_dicts else []
            cr_pre = evaluate_chunk_layer(true_chunks, pre_contents, k=5)
            chunk_recalls_pre.append(cr_pre["Recall@k"]); chunk_precisions_pre.append(cr_pre["Precision@k"]); chunk_mrrs_pre.append(cr_pre["MRR"])

            post_contents = [c["content"] for c in post_dicts] if post_dicts else []
            cr_post = evaluate_chunk_layer(true_chunks, post_contents, k=5)
            chunk_recalls_post.append(cr_post["Recall@k"]); chunk_precisions_post.append(cr_post["Precision@k"]); chunk_mrrs_post.append(cr_post["MRR"])

            ans = generate_answer(q, final_ctx)
            clean_ans, vp, ctx_full = verify_and_clean(ans, final_ctx, embedder)
            ans_eval = evaluate_answer(clean_ans, true_ans)
            answer_em.append(ans_eval["EM"]); answer_f1.append(ans_eval["F1"])
            bleu_scores.append(ans_eval["BLEU-1"]); rouge_scores.append(ans_eval["ROUGE-L"])

            sentences = re.split(r'(?<=[。！？.!?])\s*', clean_ans)
            valid_sents = [s.strip() for s in sentences if len(s.strip()) > 5]
            ssr = evaluate_evidence_layer(valid_sents, [c["content"] for c in ctx_full], embedder, batch_fact_check)
            ssr_scores.append(ssr)

        n = len(qa_set)
        results.append({
            "Setup": s["name"],
            "Page R@3": f"{np.mean(page_recalls):.3f}",
            "Page P@3": f"{np.mean(page_precisions):.3f}",
            "Page MRR": f"{np.mean(page_mrrs):.3f}",
            "Chunk pre R@5": f"{np.mean(chunk_recalls_pre):.3f}",
            "Chunk pre P@5": f"{np.mean(chunk_precisions_pre):.3f}",
            "Chunk pre MRR": f"{np.mean(chunk_mrrs_pre):.3f}",
            "Chunk post R@5": f"{np.mean(chunk_recalls_post):.3f}",
            "Chunk post P@5": f"{np.mean(chunk_precisions_post):.3f}",
            "Chunk post MRR": f"{np.mean(chunk_mrrs_post):.3f}",
            "Answer EM": f"{np.mean(answer_em):.3f}",
            "Answer F1": f"{np.mean(answer_f1):.3f}",
            "Answer BLEU-1": f"{np.mean(bleu_scores):.3f}",
            "Answer ROUGE-L": f"{np.mean(rouge_scores):.3f}",
            "SSR (Evidence)": f"{np.mean(ssr_scores):.3f}"
        })
    return pd.DataFrame(results)