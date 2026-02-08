import os
from dotenv import load_dotenv

load_dotenv()

# 使用国内 HF 镜像（可选，加速模型下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# LLM 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 模型名称
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

# 文本分块参数
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# 检索参数
TOP_K_VECTOR = 30
TOP_K_KEYWORD = 20

# 关键：精排后保留的 Top N（你缺失的就是这一行！）
FINAL_TOP_N = 5  # 可设为 5~10