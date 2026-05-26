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
            (final_ctx, pages, pre_contents, pre_scores,
             post_contents, post_scores, pre_dicts, post_dicts) = retrieve_hierarchical(
                q, p_tc, p_vc, t_c, v_c, embedder, reranker,
                use_page=s["use_page"], use_vis=s["use_vis"],
                dynamic_weight=s["dynamic"], return_trace=True
            )

            # ---------- 检索层评估 ----------
            pr = evaluate_retrieval_layer(true_pages, pages, k=3)
            page_recalls.append(pr["Recall@k"]); page_precisions.append(pr["Precision@k"]); page_mrrs.append(pr["MRR"])

            pre_contents_ = [c["content"] for c in pre_dicts] if pre_dicts else []
            cr_pre = evaluate_chunk_layer(true_chunks, pre_contents_, k=5)
            chunk_recalls_pre.append(cr_pre["Recall@k"]); chunk_precisions_pre.append(cr_pre["Precision@k"]); chunk_mrrs_pre.append(cr_pre["MRR"])

            post_contents_ = [c["content"] for c in post_dicts] if post_dicts else []
            cr_post = evaluate_chunk_layer(true_chunks, post_contents_, k=5)
            chunk_recalls_post.append(cr_post["Recall@k"]); chunk_precisions_post.append(cr_post["Precision@k"]); chunk_mrrs_post.append(cr_post["MRR"])

            # ---------- 答案生成 + 回退 ----------
            ans = generate_answer(q, final_ctx)
            clean_ans, vp, ctx_full = verify_and_clean(ans, final_ctx, embedder)

            if not clean_ans or len(clean_ans.strip()) < 3:
                clean_ans = ans
                vp = list(set(c["page"] for c in final_ctx))

            # ---------- 调试打印 ----------
            print(f"\n{'='*60}")
            print(f"Setup: {s['name']}")
            print(f"Q: {q}")
            print(f"True Answer: {true_ans[:80]}")
            print(f"Generated Answer: {ans[:80]}")
            print(f"Cleaned Answer: {clean_ans[:80]}")
            print(f"Verified Pages: {vp}")

            # ---------- 答案层评估 ----------
            ans_eval = evaluate_answer(clean_ans, true_ans)
            print(f"EM={ans_eval['EM']}, F1={ans_eval['F1']}")
            answer_em.append(ans_eval["EM"]); answer_f1.append(ans_eval["F1"])
            bleu_scores.append(ans_eval["BLEU-1"]); rouge_scores.append(ans_eval["ROUGE-L"])

            # ---------- 证据层评估 ----------
            sentences = re.split(r'(?<=[。！？.!?])\s*', clean_ans)
            valid_sents = [s.strip() for s in sentences if len(s.strip()) > 2]
            if not valid_sents:
                valid_sents = [clean_ans.strip()] if clean_ans.strip() else []
            print(f"Valid Sentences: {valid_sents}")
            ssr = evaluate_evidence_layer(valid_sents, [c["content"] for c in ctx_full], embedder, batch_fact_check)
            print(f"SSR: {ssr}")
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