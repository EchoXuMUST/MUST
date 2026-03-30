# VQA 问答对在 DeepResearch 内的运行示例与评估报告说明

本文用一个简化示例说明：

1. 单条 VQA 样本如何在 `eval_vqa.py` 中被处理；
2. 为什么会得到 baseline / graph 两套结果；
3. 报告中的关键字段如何解读。

---

## 1) 示例输入（概念化）

假设样本（MMMU/M3COT/A-OKVQA 归一化后）为：

- `question`: 图中哪个选项正确？
- `choices`: [A, B, C, D]
- `answer`: C（在归一化后会映射成选项文本）
- `documents`: 一组辅助文本
- `image`: 一张图片（可能是路径、URL 或 bytes）

程序内部会生成统一 sample，确保后续流程不依赖原始数据源字段差异。

---

## 2) 单题运行路径

### Step A: 图像解析

`resolve_image_reference()` 会按顺序尝试：

1. `image_url`
2. `__image_bytes`
3. `image_path/image/img_path` + `--image-root`

成功后得到可发送给多模态接口的 `image_ref`（常为 data URL）。

### Step B: 构造两类上下文

- **baseline_context**：直接拼接原始文档前几条
- **graph_context**：用 lexical graph 检索后拼接命中片段

这就是为什么评估会有两条预测链路：baseline vs graph。

### Step C: 两次推理

对同一问题会调用两次 `answer_with_context()`：

1. `baseline_pred`
2. `graph_pred`

若任一次异常，会记录到 `baseline_error` / `graph_error`，同时该次预测置空字符串，流程继续。

### Step D: 评分

- 精确匹配：`baseline_exact` / `graph_exact`
- 字符级 F1：`baseline_f1` / `graph_f1`
- 难度分项：`difficulty.*`

并记录样本追踪键 `sample_key`，用于断点续跑去重。

---

## 3) 报告中关键字段解释

### 3.1 全局指标

- `baseline_exact`, `graph_exact`：整体准确率
- `baseline_f1`, `graph_f1`：整体 F1
- `baseline_error_rate`, `graph_error_rate`：调用异常比例

### 3.2 难度指标

- `difficulty_scores.overall`：难度均值
- `difficulty_scores.*`：各分项均值
- `difficulty_bucket_thresholds.q33/q67`：三等分分位数阈值

### 3.3 分桶指标

`metrics_by_difficulty_bucket` 中按 `easy/medium/hard` 分组统计：

- 样本数 `count`
- 两条链路准确率/F1
- 桶内平均难度 `avg_difficulty`

---

## 4) 断点续跑示例

首次运行：

```bash
python eval_vqa.py data.jsonl \
  --details-out details.jsonl \
  --details-format jsonl \
  --save-every 100
```

中断后恢复：

```bash
python eval_vqa.py data.jsonl \
  --details-out details_resume.jsonl \
  --details-format jsonl \
  --resume-from details.jsonl
```

恢复时会读取已有 `sample_key` 并跳过已完成样本。

---

## 5) 常见现象说明

1. **有些题 200，有些题 400/500**
   - 多见于图像输入异常或服务侧资源波动；
   - 评估脚本不会崩，会把错误写入 `*_error` 字段。

2. **baseline 和 graph 同时报错**
   - 常见于服务端当前不可用或该样本图像请求体有问题；
   - 因为同一题会发两次请求，所以两条链路可能同时失败。

3. **难度桶分布不固定**
   - 当前是按当次数据分布的 q33/q67 动态分桶，不是固定阈值。

---

## 6) 小结

通过统一归一化 + 双上下文推理 + 分位数分桶 + 可恢复执行，项目可以在多数据集场景下稳定地产生可分析、可追踪的 VQA 评估报告。
