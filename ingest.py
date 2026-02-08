import os
import glob
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from modules.vector_search import QdrantVectorStore
from modules.es_search import ESSearcher

def read_docs(folder):
    texts = []
    for file in glob.glob(os.path.join(folder, "*")):
        print(f"正在解析: {file}")
        elements = partition(filename=file)
        text = "\n".join([str(e) for e in elements])
        texts.append(text)
    return texts

def chunk_text(text, chunk_size=512, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

if __name__ == "__main__":
    # 1. 读取文档
    docs = read_docs("sample_docs")
    
    # 2. 分块
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    print(f"共生成 {len(all_chunks)} 个文本块")

    # 3. 导入 Qdrant
    print("正在导入 Qdrant...")
    vs = QdrantVectorStore()
    vs.add_texts(all_chunks)

    # 4. 导入 Elasticsearch
    print("正在导入 Elasticsearch...")
    es = ESSearcher()
    es.add_texts(all_chunks)

    print("知识库导入完成！")