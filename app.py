import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import fitz
import tempfile
import ast
import cv2
import numpy as np
import json
import shutil
import sys
import pandas as pd
from pathlib import Path

# 本地模块
from config import *
from ocr_utils import setup_ocr
from embedder import load_models, UnifiedEmbedder
from utils import TempImageManager, get_file_hash
from layout_parser import extract_layout_structured
from indexer import get_chroma, build_index
from retriever import retrieve_hierarchical
from generator import generate_answer, verify_and_clean
from ablation import run_ablation_study
from qa_generator import smart_auto_generate_qa, auto_generate_qa_with_chunks

# ---- 初始设置 ----
tesseract_ok = setup_ocr()
st_model, clip_model, clip_proc, reranker, device = load_models()
embedder = UnifiedEmbedder(st_model, clip_model, clip_proc, device)
img_manager = TempImageManager()

# 初始化会话状态
if "all_data" not in st.session_state:
    st.session_state.all_data = {}
if "hash" not in st.session_state:
    st.session_state.hash = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "hist" not in st.session_state:
    st.session_state.hist = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# ---- UI ----
st.set_page_config(layout="wide", page_title="多模态RAG评估系统")
st.title("📚 多模态文档问答（智能自动生成测试集）")

col1, col2, col3 = st.columns([1, 1.8, 1.2])

with col1:
    st.header("📁 文档管理")
    up = st.file_uploader("上传 PDF", type="pdf")
    if st.button("🗑️ 清空会话", use_container_width=True):
        st.session_state.clear()
        st.session_state.all_data = {}
        st.session_state.hash = None
        st.session_state.pdf_path = None
        st.session_state.hist = []
        st.session_state.last_answer = ""
        st.session_state.last_question = ""
        st.session_state.feedback = []
        img_manager.cleanup()
        try:
            client = get_chroma()
            for name in ["page_text_idx","page_visual_idx","text_idx","visual_idx"]:
                client.delete_collection(name)
        except:
            pass
        st.rerun()

    if up:
        hb = get_file_hash(up.read())
        up.seek(0)
        if st.session_state.get("hash") != hb:
            with st.spinner("🧠 版面分析 + 双通道索引..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.read())
                    tmp.flush()
                    doc = fitz.open(tmp.name)
                    data = {}
                    for i, p in enumerate(doc):
                        data[i+1] = extract_layout_structured(
                            p, i+1, zoom=2, img_manager=img_manager,
                            clip_proc=clip_proc, clip_model=clip_model,
                            device=device, tesseract_available=tesseract_ok
                        )
                    doc.close()
                    img_manager.init_dir()
                    build_index(data, embedder, img_manager)
                    st.session_state.all_data = data
                    st.session_state.pdf_path = tmp.name
                    st.session_state.hash = hb
            st.success("✅ 索引就绪")

    st.divider()
    st.subheader("📊 消融实验（智能自动生成 + 人工标注优先）")

    if st.button("🚀 运行消融实验", use_container_width=True):
        if not st.session_state.get("hash"):
            st.warning("请先上传PDF")
            st.stop()

        p_tc = get_chroma().get_collection("page_text_idx")
        p_vc = get_chroma().get_collection("page_visual_idx")
        t_c = get_chroma().get_collection("text_idx")
        v_c = get_chroma().get_collection("visual_idx")

        # 加载或生成测试集
        qa_path = f"data/test_qa_{st.session_state.hash}.json"
        if os.path.exists("data/test_qa.json"):
            with open("data/test_qa.json", "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            qa_list = [(q["question"], q["answer"], q["pages"], q["evidence_chunks"]) for q in qa_data]
            st.success("✅ 已加载人工标注测试集")
        elif os.path.exists(qa_path):
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            qa_list = [(q["question"], q["answer"], q["pages"], q["evidence_chunks"]) for q in qa_data]
            st.success(f"✅ 已加载该文档的自动生成测试集（{len(qa_list)} 条）")
        else:
            with st.spinner("🤖 正在智能生成测试集（调用大模型改写问题，减少数据污染）..."):
                qa_list = smart_auto_generate_qa(st.session_state.all_data, max_per_type=10)
                if qa_list:
                    qa_data = [{"question": q, "answer": a, "pages": p, "evidence_chunks": c} for q,a,p,c in qa_list]
                    os.makedirs("data", exist_ok=True)
                    with open(qa_path, "w", encoding="utf-8") as f:
                        json.dump(qa_data, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ 已自动生成并保存 {len(qa_list)} 条问答")
                else:
                    qa_list = auto_generate_qa_with_chunks(st.session_state.all_data, num_text=3, num_table=2, num_figure=1)
                    st.warning("⚠️ 智能生成失败，使用简单自动生成（指标虚高，仅作演示）")

        st.info(f"📋 测试集规模：{len(qa_list)} 条")
        with st.spinner("🧪 运行消融实验..."):
            res_df = run_ablation_study(qa_list, p_tc, p_vc, t_c, v_c, embedder, reranker)
        st.dataframe(res_df, use_container_width=True)

        st.subheader("📈 关键指标对比")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("最佳页面召回 (R@3)", res_df.iloc[-1]["Page R@3"])
        with col_b:
            st.metric("最佳答案 F1", res_df.iloc[-1]["Answer F1"])
        with col_c:
            st.metric("最佳证据支持率 (SSR)", res_df.iloc[-1]["SSR (Evidence)"])

    st.divider()
    st.subheader("📈 端到端可用性评价")
    with st.expander("提交人工评价", expanded=False):
        col_score, col_safety = st.columns(2)
        with col_score:
            score = st.slider("满意度（1-5）", 1, 5, 3, key="satisfaction")
        with col_safety:
            safety = st.radio("是否安全合规？", ["是", "否（包含错误或有害内容）"], key="safety")
        if st.button("提交评价", use_container_width=True):
            if st.session_state.last_answer:
                st.session_state.feedback.append({
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "score": score,
                    "safety": safety
                })
                st.success(f"已记录评价（共 {len(st.session_state.feedback)} 条）")
            else:
                st.warning("请先生成一次回答")

with col2:
    st.header("💬 智能问答")
    for m in st.session_state.get("hist", []):
        st.chat_message(m["role"]).write(m["content"])

    if q := st.chat_input("请输入问题..."):
        if not st.session_state.get("hash"):
            st.error("请先上传PDF")
            st.stop()
        st.chat_message("user").write(q)
        st.session_state.last_question = q
        with st.spinner("🔍 双通道检索 + 重排序 + 事实核查..."):
            p_tc = get_chroma().get_collection("page_text_idx")
            p_vc = get_chroma().get_collection("page_visual_idx")
            t_c = get_chroma().get_collection("text_idx")
            v_c = get_chroma().get_collection("visual_idx")

            # 调用检索并获取 trace
            (result_ctx, pages, pre_contents, pre_scores,
             post_contents, post_scores, pre_dicts, post_dicts) = retrieve_hierarchical(
                q, p_tc, p_vc, t_c, v_c, embedder, reranker,
                use_page=True, use_vis=True, dynamic_weight=True, return_trace=True
            )

            ans = generate_answer(q, result_ctx)
            clean_ans, vp, ctx_full = verify_and_clean(ans, result_ctx, embedder)

            # 修复1：如果清理后的答案为空，回退到原始答案
            if not clean_ans or len(clean_ans.strip()) < 3:
                clean_ans = ans
                vp = list(set(c["page"] for c in result_ctx))  # 使用所有上下文页码

            st.session_state.last_answer = clean_ans

            # 保存到历史，附上 trace
            hist_entry = {
                "role": "assistant",
                "content": clean_ans,
                "ctx": ctx_full,          # 使用验证后的上下文（即result_ctx）
                "vp": vp,
                "trace": {
                    "pages": pages,
                    "pre_chunks": pre_contents,
                    "pre_scores": pre_scores,
                    "post_chunks": post_contents,
                    "post_scores": post_scores,
                    "question": q
                }
            }
            hist = st.session_state.get("hist", [])
            hist.extend([{"role": "user", "content": q}, hist_entry])
            st.session_state.hist = hist
            st.rerun()

with col3:
    st.header("🔍 证据溯源")
    if st.session_state.get("hist") and st.session_state.hist[-1]["role"] == "assistant":
        last = st.session_state.hist[-1]
        st.success(f"✅ 已验证页码: {last['vp']}")

        # 修复2：如果没有引用页码，展示所有上下文（防止溯源空白）
        display_ctx = [c for c in last["ctx"] if c["page"] in last["vp"]] if last["vp"] else last["ctx"]

        for c in display_ctx:
            st.subheader(f"📄 第 {c['page']} 页 | {c['src']} | 相似度 {c['score']:.2f}")
            if st.session_state.pdf_path:
                doc = fitz.open(st.session_state.pdf_path)
                pg = doc[c["page"] - 1]
                pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                try:
                    bbox = ast.literal_eval(c.get("bbox_str", "(0,0,0,0)"))
                except Exception:
                    bbox = (0, 0, 0, 0)
                x0, y0, x1, y1 = bbox
                if x0 != 0 or y0 != 0 or x1 != 0 or y1 != 0:
                    z = 2
                    cv2.rectangle(img, (int(x0 * z), int(y0 * z)), (int(x1 * z), int(y1 * z)), (0, 255, 0), 4)
                st.image(img)
                with st.expander("📜 证据原文", expanded=False):
                    st.text(c["content"][:400])
                doc.close()
        st.divider()

        # 检索过程可视化
        trace = last.get("trace")
        if trace:
            with st.expander("🔬 检索过程可视化（点击展开）"):
                st.subheader("📄 页面粗筛结果")
                if trace["pages"]:
                    st.markdown(f"候选页：{', '.join(f'第{p}页' for p in trace['pages'])}")
                else:
                    st.info("无候选页")

                col_pre, col_post = st.columns(2)
                with col_pre:
                    st.subheader("⚡ Chunk 得分（重排序前）")
                    if trace["pre_chunks"]:
                        df_pre = pd.DataFrame({
                            "得分": trace["pre_scores"],
                            "内容": [c[:60] + "..." for c in trace["pre_chunks"]]
                        })
                        st.dataframe(df_pre, height=250)
                        st.bar_chart(df_pre.set_index("内容")["得分"])
                    else:
                        st.info("无预排序chunk")

                with col_post:
                    st.subheader("🎯 Chunk 得分（重排序后）")
                    if trace["post_chunks"]:
                        df_post = pd.DataFrame({
                            "得分": trace["post_scores"],
                            "内容": [c[:60] + "..." for c in trace["post_chunks"]]
                        })
                        st.dataframe(df_post, height=250)
                        st.bar_chart(df_post.set_index("内容")["得分"])
                    else:
                        st.info("无重排序chunk")

                # Embedding 空间可视化
                if st.button("🧮 生成 Embedding 空间可视化（PCA）"):
                    with st.spinner("正在计算降维..."):
                        try:
                            from sklearn.decomposition import PCA
                            import matplotlib.pyplot as plt

                            q_vec_st = embedder.encode_text_st([last.get("trace", {}).get("question", "")])[0]
                            chunk_vecs = embedder.encode_text_st(trace["pre_chunks"])

                            all_vecs = np.vstack([q_vec_st, chunk_vecs])
                            labels = ["查询"] + [f"Chunk{i+1}" for i in range(len(trace["pre_chunks"]))]

                            pca = PCA(n_components=2)
                            reduced = pca.fit_transform(all_vecs)

                            fig, ax = plt.subplots()
                            ax.scatter(reduced[1:, 0], reduced[1:, 1], c='blue', label='文档块')
                            ax.scatter(reduced[0, 0], reduced[0, 1], c='red', marker='*', s=200, label='查询')
                            for i, label in enumerate(labels):
                                ax.annotate(label, (reduced[i, 0], reduced[i, 1]))
                            ax.legend()
                            st.pyplot(fig)
                        except ImportError:
                            st.error("需要安装 scikit-learn 和 matplotlib")