"""CSQA 评估项目的默认配置。"""
from pathlib import Path

# 数据与资源的默认路径，可根据实际环境覆盖。
DATA_FILE = Path("/data/zhuxy/office/dataset/NOKG_QA/CommensenseQA/data/train-00000-of-00001.parquet")
CONCEPTNET_FILE = Path("/data/zhuxy/office/dataset/NOKG_QA/CommensenseQA/data/resources/conceptnet/conceptnet-assertions-5.7.0.csv")

# 句向量与相似度模型路径
SENTENCE_EMBEDDING_MODEL = Path("/data/zhuxy/office/all-MiniLM-L6-v2")
MPNET_EMBEDDING_MODEL = Path("/data/zhuxy/office/all-mpnet-base-v2")

# NLI 模型路径
NLI_MODEL = Path("/data/zhuxy/office/bart-large-mnli")

# 安全检测相关模型路径
TOXIC_MODEL = Path("/data/zhuxy/office/toxic-bert")
UNBIASED_TOXIC_MODEL = Path("/data/zhuxy/office/unbiased-toxic-roberta")

# 敏感词表路径
SENSITIVE_TERMS_FILE = Path(__file__).resolve().parent / "sensitive_terms_zh_en.txt"

# 输出目录
OUTPUT_DIR = Path("output/csqa_assessment")
