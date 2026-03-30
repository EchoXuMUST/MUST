# DeepResearch-Qwen3VL（GraphRAG-Lexical 深化版）

这是一个参考 Tongyi-Qwen-DeepResearch 思路、并将基座模型统一为 **Qwen3-VL-8B-Instruct** 的深度研究框架。

本次进一步引入了来自 `awslabs/graphrag-toolkit/lexical-graph` 的关键思想：
- 分层 lexical graph（source/chunk/entity）
- 检索时采用“种子召回 + 图遍历扩展”
- 用于跨文档主题关联与证据补全

> 目标：让 deepresearch 的检索部分从“单跳网页抓取”升级为“知识增强检索”，并支持多模态 VQA 评测。

## 方法理解与接入位置

根据 lexical-graph 的 querying 机制，可抽象为：
1. 向量/词法相似检索得到初始 seed；
2. 从 seed 在图中沿主题/实体关系遍历，补全相关证据；
3. 用聚合结果进行答案生成。

在本项目中的对应接入：
- 在 `open_webpage` 后，将网页文本 chunk 化，构建轻量 lexical graph；
- 新增 `retrieve_lexical_graph` 工具执行“词法 seed + 实体扩展”检索；
- `ResearcherAgent` 在每个子问题结束后自动追加一次 lexical-graph 检索，补充跨文档证据；
- 再进入 `Critic/Verifier/Synthesizer` 闭环。

## 当前架构

- `PlannerAgent`: 子问题分解（结构化）
- `ResearcherAgent`: ReAct 式工具调用 + lexical graph 补检索
- `CriticAgent`: 判断证据充分性并生成追问
- `VerifierAgent`: 证据筛选
- `SynthesizerAgent`: 结构化报告生成

## 新增能力

### 1) lexical graph 检索模块
- 文件：`lexical_graph.py`
- 关键类：
  - `LexicalGraphIndex`
  - `LexicalGraphRetriever`
- 检索策略：
  - lexical seed（token overlap + idf）
  - entity traversal expansion（实体邻接扩展）
  - rerank（融合 lexical/实体匹配）

### 2) ToolRouter 新工具
- `retrieve_lexical_graph {question, documents, top_k}`

### 3) 知识增强型多模态 VQA 评测脚本
- 文件：`eval_vqa.py`
- 对比：
  - baseline（直接拼接文档上下文）
  - graph-enhanced（lexical graph 检索上下文）
- 指标：
  - Exact Match
  - 字符级 F1

## 目录结构

```text
deepresearch_qwen3_vl/
├── agents.py
├── config.py
├── eval_vqa.py                 # 新增：VQA评测
├── lexical_graph.py            # 新增：lexical-graph 检索
├── main.py
├── models.py
├── tools.py
├── workflow.py
└── requirements.txt
```

## 运行

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置环境变量

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

export MAX_RESEARCH_ROUNDS="3"
export MAX_SNIPPETS_PER_ROUND="5"
export MAX_ACTIONS_PER_QUESTION="4"
```

### 3) deepresearch 主流程

```bash
python main.py "比较2024-2025年主流多模态大模型在文档理解上的优势与局限"
```

### 4) 运行 VQA 评测

数据格式（JSON 列表，每条包含 `question/answer/documents`，可选 `image_url` 或 `image_path/image/img_path`）：

```json
[
  {
    "question": "图中模型属于哪一类？",
    "image_url": "https://.../image.png",
    "answer": "多模态大模型",
    "documents": [
      {"source": "doc1", "text": "..."},
      {"source": "doc2", "text": "..."}
    ],
    "image_path": "relative/or/absolute/path/to/image.png"
  }
]
```

运行：

```bash
python eval_vqa.py /path/to/vqa_dataset.json --out vqa_eval_report.json --image-root /path/to/images
```


支持数据集格式：
- `.json`（对象列表）
- `.jsonl`（每行一个对象）
- `.parquet`（表格列）

Parquet/自定义字段名示例：

```bash
python eval_vqa.py /data/xxx/test.parquet   --question-field question   --answer-field answer   --image-path-field image_path   --documents-field documents   --image-root /data/xxx/images   --out vqa_eval_report.json
```

如果你的 parquet 里上下文字段不是 `documents`，而是纯文本列（如 `context`），可用：


A-OKVQA parquet 特殊说明：
- 若 `image` 列是 `{"bytes": ..., "path": ...}` 这类 HuggingFace Image 对象，本脚本会自动将 `bytes` 临时落盘并作为 `file://` 图片输入模型。
- 该逻辑仅在检测到 `image` 字段且其结构确实包含 `bytes` 时触发；其他数据集不会强制按此方式解析。


```bash
python eval_vqa.py /data/xxx/test.parquet   --documents-field context   --question-field question   --answer-field answer
```

MMMU_Pro（题目 parquet 与视觉 parquet 分离）说明：
- `--dataset-type mmmu_pro`：强制走 MMMU_Pro 解析，不影响 A_OKVQA / M3COT 分支。
- `--mmmu-vision-parquet`：传入视觉 parquet 文件或目录（按 `id` 做 join，可通过 `--mmmu-id-field` / `--mmmu-vision-id-field` 改字段名）。
- `options/choices` 支持 parquet list，也支持字符串形式（如 `"['A','B',...]"`）。
- 4 选项和 10 选项都使用同一解析逻辑：模型提示只会按实际选项数量生成 A-D 或 A-J。

## 3) 运行评测

> `run_deepresearch.sh eval` 会把参数原样传给 `eval_vqa.py`，所以两者命令等价。  
> 注意：`run_deepresearch.sh` 里 `eval` 模式是**位置参数 dataset**（不是 `--dataset`）。

### 3.00) 多模型切换（Qwen3-VL / Qwen2.5-VL / Qwen2-VL）

评测命令可直接加 `--model-name` 指定模型路径（同一套数据集命令可复用）：

```bash
--model-name /data/zhuxy/multimodal_deepresearch/Qwen3-VL-8B-Instruct
--model-name /data/zhuxy/multimodal_deepresearch/Qwen2.5-VL-7B-Instruct
--model-name /data/zhuxy/multimodal_deepresearch/Qwen2-VL-7B-Instruct
```

如果你连接不同推理服务，也可附加：
- `--openai-base-url ...`
- `--openai-api-key ...`

### 3.0) 消融装配开关（按图片里的 7 种设置）

新增 `--ablation-profile`，用于控制推理时是否装配：
- KG-RAG
- 难度评估器
- 首轮推理后的 SFT 精炼

可选值与图片对应关系：
- `none`：KG-RAG ✗ / 难度评估器 ✗ / SFT ✗
- `kg_only`：KG-RAG ✓ / 难度评估器 ✗ / SFT ✗
- `difficulty_only`：KG-RAG ✗ / 难度评估器 ✓ / SFT ✗
- `kg_difficulty`：KG-RAG ✓ / 难度评估器 ✓ / SFT ✗
- `kg_sft`：KG-RAG ✓ / 难度评估器 ✗ / SFT ✓
- `difficulty_sft`：KG-RAG ✗ / 难度评估器 ✓ / SFT ✓
- `all_on`：KG-RAG ✓ / 难度评估器 ✓ / SFT ✓

示例（只做 KG-RAG，不用难度评估和 SFT）：

```bash
bash run_deepresearch.sh eval \
  /path/to/dataset.json \
  --dataset-type generic \
  --ablation-profile kg_only \
  --out /tmp/ablation_kg_only.json
```

评测结束后，终端会额外打印 `=== Accuracy Stats ===`，并在输出 JSON 中保存：
- `accuracy_stats.total`
- `accuracy_stats.baseline_correct`
- `accuracy_stats.graph_correct`
- `accuracy_stats.baseline_accuracy`
- `accuracy_stats.graph_accuracy`

### 3.1) A_OKVQA

```bash
bash run_deepresearch.sh eval \
  /data/zhuxy/office/dataset/KG_VQA/A_OKVQA/test-00000-of-00001-d306bf3ad53b6618.parquet \
  --dataset-type generic \
  --question-field question \
  --answer-field answer \
  --image-path-field image_path \
  --documents-field documents \
  --image-root /data/zhuxy/multimodal_deepresearch/A_OKVQA/images/train2017 \
  --out /data/zhuxy/multimodal_deepresearch/output/aokvqa_eval_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/aokvqa_eval_details.jsonl \
  --details-format jsonl
```

```bash
# 等价 Python 版本
python eval_vqa.py \
  /data/zhuxy/multimodal_deepresearch/A_OKVQA/train-00000-of-00002-c1d24de3bacb5e0c.parquet \
  --dataset-type generic \
  --question-field question \
  --answer-field answer \
  --documents-field documents \
  --image-path-field image_path \
  --image-root /data/zhuxy/multimodal_deepresearch/A_OKVQA/images/train2017 \
  --out /data/zhuxy/multimodal_deepresearch/output/a_okvqa_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/a_okvqa_details.jsonl \
  --details-format jsonl
```

### 3.2) M3COT

```bash
bash run_deepresearch.sh eval \
  /data/zhuxy/multimodal_deepresearch/M3COT/data/train.jsonl \
  --dataset-type m3cot \
  --question-field question \
  --answer-field answer \
  --image-path-field image \
  --image-root /data/zhuxy/multimodal_deepresearch/M3COT/data/images \
  --out /data/zhuxy/multimodal_deepresearch/output/m3cot_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/m3cot_details.jsonl \
  --details-format jsonl
```

```bash
# 等价 Python 版本
python eval_vqa.py \
  /data/zhuxy/multimodal_deepresearch/M3COT/data/train.jsonl \
  --dataset-type m3cot \
  --question-field question \
  --answer-field answer \
  --image-path-field image \
  --image-root /data/zhuxy/multimodal_deepresearch/M3COT/data/images \
  --out /data/zhuxy/multimodal_deepresearch/output/m3cot_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/m3cot_details.jsonl \
  --details-format jsonl
```

### 3.3) MMMU_pro（4 options）

```bash
bash run_deepresearch.sh eval \
  /data/zhuxy/multimodal_deepresearch/MMMU_pro/standard_4 \
  --dataset-type mmmu_pro \
  --question-field question \
  --answer-field answer \
  --mmmu-vision-parquet /data/zhuxy/multimodal_deepresearch/MMMU_pro/vision \
  --mmmu-id-field id \
  --mmmu-vision-id-field id \
  --mmmu-vision-image-field image \
  --out /data/zhuxy/multimodal_deepresearch/output/mmmu_pro_4_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/mmmu_pro_4_details.jsonl \
  --details-format jsonl
```

```bash
# 等价 Python 版本
python eval_vqa.py \
  /data/zhuxy/multimodal_deepresearch/MMMU_pro/standard_4 \
  --dataset-type mmmu_pro \
  --question-field question \
  --answer-field answer \
  --mmmu-vision-parquet /data/zhuxy/multimodal_deepresearch/MMMU_pro/vision \
  --mmmu-id-field id \
  --mmmu-vision-id-field id \
  --mmmu-vision-image-field image \
  --out /data/zhuxy/multimodal_deepresearch/output/mmmu_pro_4_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/mmmu_pro_4_details.jsonl \
  --details-format jsonl
```

### 3.4) MMMU_pro（10 options）

```bash
bash run_deepresearch.sh eval \
  /data/zhuxy/multimodal_deepresearch/MMMU_pro/standard_10 \
  --dataset-type mmmu_pro \
  --question-field question \
  --answer-field answer \
  --mmmu-vision-parquet /data/zhuxy/multimodal_deepresearch/MMMU_pro/vision \
  --mmmu-id-field id \
  --mmmu-vision-id-field id \
  --mmmu-vision-image-field image \
  --out /data/zhuxy/multimodal_deepresearch/output/mmmu_pro_10_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/mmmu_pro_10_details.jsonl \
  --details-format jsonl
```

### 3.5) CMMQA

```bash
bash run_deepresearch.sh eval \
  /data/zhuxy/multimodal_deepresearch/problems_with_urls.json \
  --dataset-type cmmqa \
  --question-field question \
  --answer-field answer \
  --image-url-field image_url \
  --image-path-field image \
  --image-root /data/zhuxy/multimodal_deepresearch \
  --out /data/zhuxy/multimodal_deepresearch/output/cmmqa_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/cmmqa_details.jsonl \
  --details-format jsonl
```

### 3.6) ScienceQA

> `problems.json` 为 `dict[id -> problem]` 结构；脚本会自动展开并把 `id` 注入样本。  
> 图像路径按 `images/{split}/{id}/image.png` 规则拼接（例如 `train/1/image.png`）。

```bash
bash run_deepresearch.sh eval \
  /data/zhuxy/office/dataset/NOKG_VQA/ScienceQA/data/scienceqa/problems.json \
  --dataset-type scienceqa \
  --question-field question \
  --answer-field answer \
  --image-path-field image \
  --image-root /data/zhuxy/office/dataset/NOKG_VQA/ScienceQA/images \
  --out /data/zhuxy/multimodal_deepresearch/output/scienceqa_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/scienceqa_details.jsonl \
  --details-format jsonl
```

```bash
# 等价 Python 版本
python eval_vqa.py \
  /data/zhuxy/office/dataset/NOKG_VQA/ScienceQA/data/scienceqa/problems.json \
  --dataset-type scienceqa \
  --question-field question \
  --answer-field answer \
  --image-path-field image \
  --image-root /data/zhuxy/office/dataset/NOKG_VQA/ScienceQA/images \
  --out /data/zhuxy/multimodal_deepresearch/output/scienceqa_report.json \
  --details-out /data/zhuxy/multimodal_deepresearch/output/scienceqa_details.jsonl \
  --details-format jsonl
```

## 数据集题目难度评估（AOKVQA / MMMU_Pro / M3CoT 风格整合）

已将“题目难度评估”整合进 `eval_vqa.py`，并新增 `difficulty.py` 统一计算以下维度：

- `linguistic_complexity`：问题文本复杂度
- `reasoning_complexity`：多跳/链式推理复杂度
- `multimodal_complexity`：图像/表格/图表带来的复杂度
- `knowledge_intensity`：外部知识与文档依赖强度
- `choice_ambiguity`：选项歧义度（适配多选题）
- `retrieval_hardness`：lexical-graph 检索难度（DeepResearch 可评估部分）
- `deepresearch_coverage`：DeepResearch 检索覆盖度（DeepResearch 可评估部分）

并输出：
- 综合难度分 `overall`
- 难度分桶指标（easy/medium/hard）下的 baseline vs graph-enhanced EM/F1
- 全量难度维度均值（用于最终报告）

> 说明：你提供的 3 个 chatgpt.com/codex/tasks 链接在当前环境返回 403，无法直接读取原页面内容；本实现采用与这类评测常见流程一致的可复现统一框架，并预留字段（如 `reasoning_hops`/`cot_steps`/`choices`/`domain`）用于与具体数据集标注对齐。


## 一键启动脚本（bash xxx.sh）

已提供统一脚本：`run_deepresearch.sh`，可直接使用：

```bash
bash run_deepresearch.sh --help
```

### 1) 使用 DashScope API 调用（Qwen3-VL）

```bash
export DASHSCOPE_API_KEY="sk-xxxx"
bash run_deepresearch.sh dashscope "比较 Qwen3-VL 与其他模型在文档理解上的差异"
```

脚本会自动设置：
- `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `OPENAI_API_KEY=$DASHSCOPE_API_KEY`
- `MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}`

### 2) 本地 vLLM（两张 3090，默认 8/9）部署并运行

先启动 vLLM OpenAI 服务：

```bash
bash run_deepresearch.sh vllm-start-8b /path/to/Qwen3-VL-8B-Instruct
```

默认参数：
- `CUDA_VISIBLE_DEVICES_8B=8,9`
- `VLLM_TP_SIZE_8B=2`
- `model_path` 优先使用命令行参数；未传时回退到 `MODEL_NAME_8B`
- `VLLM_PORT=8000`

再在另一个终端运行 DeepResearch 查询：

```bash
bash run_deepresearch.sh vllm-run-8b "帮我总结多模态文档理解路线" /path/to/Qwen3-VL-8B-Instruct
```

### 3) 运行评测

```bash
bash run_deepresearch.sh eval --dataset /path/to/dataset.json --out /path/to/report.json --mock
```


### 2.1) 使用 Qwen3-VL-8B-Instruct 在两张卡（默认 8,9）TP 部署（唯一保留的本地部署方案）

脚本已内置 8B 快捷模式：

```bash
# 默认模型 Qwen/Qwen3-VL-8B-Instruct，默认 GPU=8,9，TP=2
bash run_deepresearch.sh vllm-start-8b

# 或指定本地权重路径
bash run_deepresearch.sh vllm-start-8b /data/models/Qwen3-VL-8B-Instruct

# 在另一个终端发起查询
bash run_deepresearch.sh vllm-run-8b "请分析这组图文问答样例"
```

可选环境变量：
- `CUDA_VISIBLE_DEVICES_8B`（默认 `8,9`）
- `VLLM_TP_SIZE_8B`（默认 `2`）
- `MODEL_NAME_8B`（默认 `Qwen/Qwen3-VL-8B-Instruct`）


稳定性说明（针对你遇到的 `Connection reset by peer`）：
- 评测时本地图片不再使用 `file://` 直接传给模型，而是转换为 `data:image/...;base64,...` 再发送，降低 vLLM 侧解析/访问异常概率。
- `models.py` 增加了 API 连接重试与超时配置（`API_TIMEOUT_S`、`API_RETRIES`、`API_RETRY_BACKOFF_S`）。
- `eval_vqa.py` 现在对单条样本调用异常做了容错，不会整批任务直接中断，报告里会记录 `baseline_error` / `graph_error`。
