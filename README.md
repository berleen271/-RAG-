# 基于分层索引与双模态表示增强的多模态RAG在复杂文档问答中的应用研究

> 110实验室前沿技术分组研究课题 | 综合应用类项目  
> 面向复杂PDF文档（文本 + 表格 + 图像）的多模态检索增强生成系统

---

## 📌 项目概述

### 课题信息

| 项目       | 内容                                                         |
| ---------- | ------------------------------------------------------------ |
| 课题名称   | 基于分层索引与双模态表示增强的多模态RAG在复杂文档问答中的应用研究 |
| 项目类型   | 综合应用类                                                   |
| 研究方向   | 多模态RAG（Multimodal Retrieval-Augmented Generation）        |
| 应用场景   | 复杂文档智能问答                                             |
| 开发形式   | Web交互式系统                                                |
| 前端框架   | Streamlit                                                    |

项目针对传统文本RAG在复杂PDF场景中的局限，提出一种基于：

- 分层索引（页级→块级）
- 双模态表示增强（文本+视觉）
- CrossEncoder重排序
- 答案事实校验

的多模态RAG方案。

---

## 🎯 研究背景与研究目标

### 研究背景

随着大模型技术的发展，RAG（Retrieval-Augmented Generation）已成为提升知识问答能力的重要方法。

但在复杂文档场景（论文、财务报表、技术文档、法律文件）中仍存在：

**① 检索粒度单一**  
传统RAG：文档 → Chunk → 向量检索，存在跨页信息缺失、长文召回能力弱、局部上下文丢失的问题。

**② 图文信息割裂**  
复杂PDF包含文本、表格、图像、图表、版面结构，传统文本RAG无法有效处理图表趋势分析、表格数据问答、图文联合理解。

**③ 生成结果可解释性不足**  
现有方案普遍存在幻觉问题、无法定位来源、缺乏事实验证。

### 研究目标

1. 构建复杂文档多模态RAG框架，支持文本、表格、图像、图表的统一处理。  
2. 设计分层检索机制（页级检索→块级检索→重排序），提高召回率。  
3. 实现双模态表示增强，采用 Sentence-BERT（文本）和 CLIP（视觉）构建统一表示空间。  
4. 实现可溯源问答，生成答案自动关联页码、图号、表号。  
5. 完成可运行系统，输出 Web 演示系统、实验报告和消融实验分析。

---

## 🔍 研究意义

**理论意义**  
- 推动RAG向多模态方向发展  
- 验证分层检索有效性  
- 探索图文联合检索机制  

**实践意义**  
应用于学术论文分析、财务报表理解、法律文书问答、技术手册检索等场景。

---

## 🚀 技术路线

### 整体Workflow

```
PDF上传
    ↓
版面解析 (PyMuPDF+OCR)
    ↓
语义分块 (Text/Table/Figure)
    ↓
双通道编码
    ├─ SentenceTransformer
    └─ CLIP
    ↓
建立索引
    ├─ 页级索引
    └─ 块级索引
    ↓
分层检索
    ↓
CrossEncoder重排序
    ↓
Qwen答案生成
    ↓
事实核查
    ↓
答案输出+证据溯源
```

---

## ⚙️ 系统架构

| 层级         | 核心组件                 | 功能                   |
| ------------ | ------------------------ | ---------------------- |
| 文档解析层   | PyMuPDF + OCR            | 提取文本、表格、图像   |
| 表示层       | SentenceTransformer + CLIP | 文本与视觉编码         |
| 索引层       | ChromaDB                 | 页级与块级索引         |
| 检索层       | Hierarchical Retrieval   | 多阶段检索             |
| 重排序层     | BAAI-reranker            | 相关性排序             |
| 生成层       | Qwen2.5                  | 答案生成               |
| 验证层       | 事实校验模块             | 减少幻觉               |
| 展示层       | Streamlit                | 交互系统               |

---

## 🧠 核心方法设计

### 1. 文档解析
使用 `extract_layout_structured()` 提取文本块、表格、图片、标题、OCR内容，输出结构化数据（含类型、内容、边界框）。

### 2. 双模态表示
- 文本：SentenceTransformer → 384维向量  
- 视觉：CLIP → 768维向量  

### 3. 分层检索
- 第一阶段：页级粗检索（Question → Page Top-3）  
- 第二阶段：Chunk精检索（Page → Chunk Top-K）  
- 第三阶段：CrossEncoder重排序  

### 4. 动态权重机制
根据问题类型调整文本与视觉权重：

| 问题类型   | 文本权重 | 视觉权重 |
| ---------- | -------- | -------- |
| 普通文本   | 0.6      | 0.4      |
| 表格问题   | 0.3      | 0.7      |
| 图表趋势   | 0.2      | 0.8      |

### 5. 事实校验
一级：LLM 事实判断；二级：余弦相似度验证。输出 Supported Sentence Ratio (SSR)。

---

## 📊 实验设计

### 数据集构建
系统自动从 PDF 构造 QA 数据，问题类型覆盖：
- 文本类：如“请解释第3页关于神经网络结构的内容”  
- 表格类：如“2023年营收是多少？”  
- 图像类：如“描述第8页图像内容”  

推荐实验规模：文本问题 50 条、表格问题 25 条、图像问题 25 条，共 100 条 QA 样本。

### 消融实验设计
四组实验，逐步添加模块：

| 实验组               | 页级检索 | 视觉检索 | 动态权重 |
| -------------------- | -------- | -------- | -------- |
| Baseline             | ×        | ×        | ×        |
| ExpA (+Hierarchical) | √        | ×        | ×        |
| ExpB (+Visual)       | √        | √        | ×        |
| ExpC (+Dynamic)      | √        | √        | √        |

目的：验证分层检索、多模态融合、动态权重的贡献。

---

## 📈 评估指标

- **页面检索层**：Recall@3, Precision@3, MRR  
- **Chunk检索层**（重排序前后）：Recall@5, Precision@5, MRR  
- **答案层**：EM (Exact Match), F1, BLEU-1, ROUGE-L  
- **证据层**：SSR (Supported Sentence Ratio)  
- **用户端到端评价**：满意度（1~5）、安全性  

### 消融实验结果示例

| Setup   | Page R@3 | Chunk post R@5 | EM   | F1   | ROUGE-L | SSR  |
| ------- | -------- | -------------- | ---- | ---- | ------- | ---- |
| Baseline| 0.45     | 0.35           | 0.10 | 0.40 | 0.35    | 0.65 |
| ExpA    | 0.62     | 0.53           | 0.30 | 0.55 | 0.52    | 0.73 |
| ExpB    | 0.70     | 0.68           | 0.45 | 0.61 | 0.60    | 0.77 |
| ExpC    | 0.75     | 0.80           | 0.52 | 0.65 | 0.68    | 0.80 |

---

## 📂 项目结构

```
project/
├── app.py
├── requirements.txt
├── environment.yml
├── chroma_db/
├── temp_images/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── test_qa.json
│   └── README.md
├── models/
│   ├── SentenceTransformer
│   ├── CLIP
│   └── CrossEncoder
├── results/
│   ├── ablation.csv
│   └── evaluation.csv
├── scripts/
│   └── preprocess.py
├── tests/
│   └── test_retrieval.py
└── README.md
```

---

## 💻 运行环境

**软件环境**  
Python 3.10, PyTorch, Transformers, SentenceTransformer, ChromaDB, Streamlit, OpenCV, PyMuPDF

**硬件环境**  
CPU：Intel i7+；GPU：RTX3060 及以上；RAM：16GB+
## 运行测试

本项目使用 pytest 进行单元测试，主要验证评估指标函数的正确性。

### 安装测试依赖
```bash
pip install pytest
---

## 🛠 快速运行

```bash
git clone https://github.com/your-name/project.git
cd project
pip install -r requirements.txt   # 或使用 conda env create -f environment.yml
streamlit run app.py
```

---
## 实验设置
- **数据集**：自建财务报告测试集，共20条问答，涵盖文本(10)、表格(5)、图表(5)三类，由人工标注，答案经过改写以减少数据污染。
- **硬件环境**：Intel i7-12700H / 16GB RAM / NVIDIA RTX3060 (6GB) / Windows 11
- **模型版本**：
  - 文本嵌入：paraphrase-multilingual-MiniLM-L12-v2
  - 视觉嵌入：openai/clip-vit-large-patch14
  - 重排序器：BAAI/bge-reranker-base
  - 生成模型：Qwen2.5-7B-Instruct (API)
  - 多模态模型：Qwen-VL-Plus (API)
- **依赖**：见 requirements.txt

## 数据来源与预处理
- **数据来源**：公开披露的贵州茅台酒股份有限公司2026年第一季度报告（PDF），包含营业收入、利润、现金流、资产负债、经销商、股东等多模态信息。
- **文档解析**：使用 PyMuPDF、pdfplumber 进行PDF文本与版面提取，PaddleOCR 处理图表区域文字。
- **表格处理**：自动提取财报中财务数据表格，转换为结构化文本与键值对。
- **图表处理**：财报中的图片、图表经 Qwen-VL 生成结构化 JSON 或文字描述，加入多模态索引。
- **分块策略**：按章节、段落、表格、图表进行分层切块，构建页级→块级分层索引。
## 🔬 数据集与可复现性

### 1. 数据集来源与构建

**文档来源**  
实验使用公开金融公告 PDF 作为测试样例，当前核心文档为：  
- 贵州茅台酒股份有限公司《2026年第一季度报告》  
- 证券代码：600519，证券简称：贵州茅台  
- 来源：上海证券交易所官网（https://www.sse.com.cn），公告日期 2026-04-25

多文档实验可扩展至同类型公告（如年度报告、半年度报告等），所有文档信息记录于 `data/raw/README.md`。

**问答对构建流程**  
采用“自动生成 + 人工校验”策略，覆盖文本、表格、图表三类问题：  
1. **候选生成**：对每个文本块、表格 Markdown、图表描述，调用 Qwen2.5 生成 2~3 个候选问题，要求答案必须能从该块中推理得到。  
2. **人工筛查**：两名研究生独立审查，删除答案模糊、跨文档或无法从单一块确定的问题，并补充边界案例。  
3. **标准答案标注**：每条数据包含问题、参考答案、证据类型（text/table/figure）、证据页码、证据片段。

最终形成的标准测试集包含 **20 条问答对**（可扩展至 100 条），分布如下：

| 类型   | 数量 | 示例问题                         |
| ------ | ---- | -------------------------------- |
| 文本类 | 12   | “一季度营业收入同比增长多少？”     |
| 表格类 | 5    | “2026年Q1归母净利润是多少？”      |
| 图表类 | 3    | “经营活动现金流净额的变动趋势如何？” |
| **总计** | 20 |                                  |

完整标注文件见 `data/test_qa.json`。

### 2. 数据预处理细节

所有预处理步骤已封装为 `scripts/preprocess.py`，可一键复现。

**2.1 文档解析**  
- 文本：PyMuPDF (fitz) 提取段落，过滤页眉页脚。  
- 表格：pdfplumber 提取表格，转为 Markdown 格式，并附带前 100 字符上下文。  
- 图表：fitz 截取图像区域，保存至 `temp_images/`，调用 glm-4v 生成结构化描述。

**2.2 文本分块**  
- 工具：LangChain RecursiveCharacterTextSplitter  
- 参数：chunk_size=512, chunk_overlap=64  
- 按页组织，保留页码元数据

**2.3 表格序列化**  
表格转为 Markdown，嵌入标题与上下文，例如：  
```markdown
## 一、主要财务数据
| 项目           | 本报告期          | 上年同期          | 增减变动幅度(%) |
|----------------|-------------------|-------------------|-----------------|
| 营业收入       | 53,909,252,220.51 | 50,600,957,885.78 | 6.54            |
| 归母净利润     | 27,242,512,886.45 | 26,847,474,238.76 | 1.47            |
上下文：该表摘自第2页，展示了公司2026年第一季度主要会计数据和财务指标...
```

**2.4 图表描述生成**  
每张图表调用多模态模型生成统一描述，格式为：  
> “该图为[图表标题]，展示了[趋势/对比/构成]，关键数据点包括：……，整体呈[上升/下降/波动]趋势。”  
描述文本存入视觉索引，与 CLIP 图像向量共同用于检索。

**2.5 向量化与索引**  
- 文本编码：`sentence-transformers/all-MiniLM-L6-v2` (384维)  
- 视觉编码：`ViT-B/32` CLIP 模型 (768维)  
- 存储引擎：ChromaDB，分别建立页级和块级 collection

### 3. 数据文件清单

```
data/
├── raw/                     # 原始 PDF（或下载说明）
├── processed/
│   ├── pages/               # 每页文本
│   ├── chunks_text.json     # 文本块列表
│   ├── tables_markdown.json # 表格数据
│   ├── charts_descriptions.json # 图表描述
│   └── metadata.json        # 文档元信息
├── test_qa.json             # 标准测试集（20 条）
└── README.md                # 数据说明
```

`test_qa.json` 样例：
```json
[
  {
    "id": 1,
    "question": "2026年第一季度营业收入同比增长多少？",
    "answer": "6.54%",
    "evidence_page": 2,
    "evidence_type": "table",
    "evidence_snippet": "营业收入 53,909,252,220.51 ... 6.54"
  }
]
```

### 4. 完整复现步骤

**4.1 环境准备**  
```bash
conda env create -f environment.yml   # 推荐
conda activate mm-rag
```
或使用 `pip install -r requirements_lock.txt`。

**4.2 数据准备与预处理**  
1. 从上海证券交易所官网下载贵州茅台《2026年第一季度报告》PDF，放入 `data/raw/` 文件夹。  
2. 运行预处理脚本：  
   ```bash
   python scripts/preprocess.py --pdf_path data/raw/贵州茅台2026年第一季度报告.pdf
   ```  
   脚本将自动完成解析、分块、向量化，并生成初始问答候选集。  
3. 人工校对 `data/test_qa_candidates.json`，保留符合要求的条目，另存为 `data/test_qa.json`。

**4.3 启动系统**  
```bash
streamlit run app.py
```  
浏览器打开 `http://localhost:8501`，上传 PDF 即可进行交互问答，并查看证据溯源与可视化。

**4.4 运行消融实验**  
```bash
python run_ablation.py --qa_file data/test_qa.json --output results/ablation.csv
```  
输出结果保存至 `results/ablation.csv`。

**4.5 使用示例数据（若无原始 PDF）**  
若因版权原因无法提供原始 PDF，项目提供脱敏示例数据：  
- `data/processed/` 下包含预处理的文本块、表格、图表描述  
- `data/test_qa.json` 可直接用于评估  
无需原始 PDF 即可运行完整的消融实验和 Streamlit 演示。

### 5. 代码健壮性说明
- 预处理脚本包含异常处理（空 PDF、损坏文件、无表格/图表等）。  
- `tests/` 目录提供基础单元测试，验证检索函数对空输入、缺失参数的正确响应。  
- 系统通过 Streamlit 界面提供手动集成测试入口，可交互式验证各模块功能。

---

## 📚 参考文献

[1] Huang et al. MDocAgent, 2025  
[2] Suri et al. VisDoM, NAACL 2025  
[3] Dong et al. Benchmarking Retrieval-Augmented Multimodal Generation, NeurIPS 2025  
[4] UNIDOC-BENCH, 2025  
[5] RAG-Anything, 2025  

---

## 📝 项目说明

本项目仅用于学术研究、教学实验和技术学习。
```
