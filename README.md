# -RAG-
基于分层索引与双模态表示增强的多模态RAG在复杂文档问答中的应用研究
[开题报告.docx](https://github.com/user-attachments/files/27899447/default.docx)
# 基于分层索引与双模态表示增强的多模态RAG在复杂文档问答中的应用研究

> 110实验室前沿技术分组研究课题 | 综合应用类项目  
> 面向复杂PDF文档（文本 + 表格 + 图像）的多模态检索增强生成系统

---

# 📌 项目概述

## 课题信息

|项目|内容|
|--|--|
|课题名称|基于分层索引与双模态表示增强的多模态RAG在复杂文档问答中的应用研究|
|项目类型|综合应用类|
|研究方向|多模态RAG（Multimodal Retrieval-Augmented Generation）|
|应用场景|复杂文档智能问答|
|开发形式|Web交互式系统|
|前端框架|Streamlit|

项目针对传统文本RAG在复杂PDF场景中的局限，提出一种基于：

- 分层索引（页级→块级）
- 双模态表示增强（文本+视觉）
- CrossEncoder重排序
- 答案事实校验

的多模态RAG方案。

---

# 🎯 研究背景与研究目标

## 研究背景

随着大模型技术的发展，RAG（Retrieval-Augmented Generation）已成为提升知识问答能力的重要方法。

但在复杂文档场景（论文、财务报表、技术文档、法律文件）中仍存在：

### ① 检索粒度单一

传统RAG：

```text
文档
↓
Chunk
↓
向量检索
```

存在：

- 跨页信息缺失
- 长文召回能力弱
- 局部上下文丢失

---

### ② 图文信息割裂

复杂PDF通常包含：

- 文本
- 表格
- 图像
- 图表
- 版面结构

传统文本RAG无法有效处理：

```text
图表趋势分析
表格数据问答
图文联合理解
```

---

### ③ 生成结果可解释性不足

现有方案普遍存在：

- 幻觉问题
- 无法定位来源
- 缺乏事实验证

---

## 研究目标

本研究目标：

### （1）构建复杂文档多模态RAG框架

支持：

- 文本
- 表格
- 图像
- 图表

统一处理。

---

### （2）设计分层检索机制

构建：

```text
页级检索
↓

块级检索
↓

重排序
```

提高召回率。

---

### （3）实现双模态表示增强

采用：

- Sentence-BERT（文本）
- CLIP（视觉）

构建统一表示空间。

---

### （4）实现可溯源问答

生成答案自动关联：

- 页码
- 图号
- 表号

---

### （5）完成可运行系统

输出：

- Web演示系统
- 实验报告
- 消融实验分析

---

# 🔍 研究意义

## 理论意义

- 推动RAG向多模态方向发展
- 验证分层检索有效性
- 探索图文联合检索机制

## 实践意义

应用于：

- 学术论文分析
- 财务报表理解
- 法律文书问答
- 技术手册检索

---

# 🚀 技术路线

## 整体Workflow

```text
PDF上传
    ↓
版面解析
(PyMuPDF+OCR)
    ↓
语义分块
(Text/Table/Figure)
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

# ⚙️ 系统架构

|层级|核心组件|功能|
|--|--:|--|
|文档解析层|PyMuPDF + OCR|提取文本、表格、图像|
|表示层|SentenceTransformer + CLIP|文本与视觉编码|
|索引层|ChromaDB|页级与块级索引|
|检索层|Hierarchical Retrieval|多阶段检索|
|重排序层|BAAI-reranker|相关性排序|
|生成层|Qwen2.5|答案生成|
|验证层|事实校验模块|减少幻觉|
|展示层|Streamlit|交互系统|

---

# 🧠 核心方法设计

## 1. 文档解析

PDF解析：

```python
extract_layout_structured()
```

提取：

- 文本块
- 表格
- 图片
- 标题
- OCR内容

输出：

```python
{
"type":"table",
"content":"...",
"bbox":...
}
```

---

## 2. 双模态表示

文本：

```python
SentenceTransformer
```

视觉：

```python
CLIP
```

向量：

```text
文本向量：384维
视觉向量：768维
```

---

## 3. 分层检索

采用：

### 第一阶段：页级粗检索

目的：

快速缩小搜索范围。

```text
Question
↓

Page Top-3
```

---

### 第二阶段：Chunk精检索

目的：

找到局部相关证据。

```text
Page
↓

Chunk Top-K
```

---

### 第三阶段：重排序

采用：

```python
CrossEncoder
```

提升最终排序质量。

---

## 4. 动态权重机制

根据问题类型调整：

|问题类型|文本权重|视觉权重|
|--|--:|--:|
|普通文本|0.6|0.4|
|表格问题|0.3|0.7|
|图表趋势|0.2|0.8|

---

## 5. 事实校验

采用：

### 一级：

LLM事实判断

### 二级：

余弦相似度验证

输出：

```text
Supported Sentence Ratio (SSR)
```

---

# 📊 实验设计

## 数据集构建

系统自动从PDF构造QA数据：

### 文本类

例：

```text
请解释第3页关于神经网络结构的内容
```

---

### 表格类

例：

```text
2023年营收是多少？
```

---

### 图像类

例：

```text
描述第8页图像内容
```

推荐实验规模：

|类别|数量|
|--|--:|
|文本问题|50|
|表格问题|25|
|图像问题|25|

总计：

```text
100 QA样本
```

---

# 🧪 消融实验设计

代码中共设计四组实验：

|实验组|页级检索|视觉检索|动态权重|
|--|--:|--:|--:|
|Baseline|×|×|×|
|ExpA (+Hierarchical)|√|×|×|
|ExpB (+Visual)|√|√|×|
|ExpC (+Dynamic Weight)|√|√|√|

目的：

验证：

- 分层检索贡献
- 多模态贡献
- 动态权重贡献

---

# 📈 评估指标

## ① 页面检索层

### Recall@3

```math
Recall@3=
\frac{|Relevant∩Retrieved|}
{|Relevant|}
```

### Precision@3

### MRR

---

## ② Chunk检索层

重排序前：

- Recall@5
- Precision@5
- MRR

重排序后：

- Recall@5
- Precision@5
- MRR

---

## ③ 答案层

### EM

Exact Match

### F1

词级匹配

### BLEU-1

### ROUGE-L

---

## ④ 证据层

SSR：

```math
SSR=
\frac{Supported\ Sentences}
{Total\ Sentences}
```

---

## ⑤ 用户端到端评价

评价内容：

- 满意度（1~5）
- 安全性

---

# 📉 消融实验结果示例

|Setup|Page R@3|Chunk post R@5|EM|F1|ROUGE-L|SSR|
|--|--:|--:|--:|--:|--:|--:|
|Baseline|0.45|0.35|0.10|0.40|0.35|0.65|
|ExpA|0.62|0.53|0.30|0.55|0.52|0.73|
|ExpB|0.70|0.68|0.45|0.61|0.60|0.77|
|ExpC|0.75|0.80|0.52|0.65|0.68|0.80|

---

# 📂 项目结构

```text
project/
│
├── app.py
├── requirements.txt
├── chroma_db/
├── temp_images/
├── data/
├── README.md
│
├── models/
│      ├── SentenceTransformer
│      ├── CLIP
│      └── CrossEncoder
│
└── results/
       ├── ablation.csv
       └── evaluation.csv
```

---

# 💻 运行环境

## 软件环境

```text
Python 3.10
PyTorch
Transformers
SentenceTransformer
ChromaDB
Streamlit
OpenCV
PyMuPDF
```

## 硬件环境

建议：

```text
CPU：Intel i7+
GPU：RTX3060及以上
RAM：16GB+
```

---

# 🛠 快速运行

```bash
git clone https://github.com/your-name/project.git

cd project

pip install -r requirements.txt

streamlit run app.py
```

---

# 📚 参考文献

[1] Huang et al. MDocAgent, 2025

[2] Suri et al. VisDoM, NAACL 2025

[3] Dong et al. Benchmarking Retrieval-Augmented Multimodal Generation, NeurIPS 2025

[4] UNIDOC-BENCH, 2025

[5] RAG-Anything, 2025

---

# 📝 项目说明

本项目仅用于：

- 学术研究
- 教学实验
- 技术学习
