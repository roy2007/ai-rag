import os
from fastapi import FastAPI
from pydantic import BaseModel
from config import FINAL_TOP_N
from modules.vector_search import QdrantVectorStore
from modules.es_search import ESSearcher
from modules.llm_qwen import qwen_intent, qwen_rewrite
from modules.rerank import BGEReranker
from modules.fusion_rank import reciprocal_rank_fusion
from modules.llm_qwen import call_qwen_for_answer  # 封装生成

app = FastAPI(title="多层RAG系统 - Win11单机版")

# 初始化检索器（启动时加载）
vector_store = QdrantVectorStore()
es_searcher = ESSearcher()
reranker = BGEReranker()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    q = request.question.strip()
    if not q:
        return {"error": "问题不能为空"}

    # 1. 意图识别
    intent = qwen_intent(q)
    if intent != "fact_query":
        return {"answer": "当前仅支持事实性问题查询。", "intent": intent}

    # 2. Query 改写（获取3个变体）
    try:
        rewritten_queries = qwen_rewrite(q)
    except Exception as e:
        rewritten_queries = [q]  # 回退

    all_results = []

    # 3. 多路召回
    for query in [q] + rewritten_queries[:2]:  # 原始 + 2个改写
        vec_res = vector_store.search(query, top_k=20)
        es_res = es_searcher.search(query, top_k=15)
        all_results.extend(vec_res + es_res)

    # 4. 去重（基于文本内容）
    seen = set()
    unique_results = []
    for r in all_results:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique_results.append(r)

    # 5. 融合排序（RRF）
    fused = reciprocal_rank_fusion(unique_results, top_k=50)

    # 6. 精排 Top 5~10
    reranked = reranker.rerank(q, fused, top_k=FINAL_TOP_N)

    # 7. 构造上下文
    context = "\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(reranked)])
    answer = call_qwen_for_answer(q, context)

    return {
        "answer": answer,
        "references": [doc["text"] for doc in reranked],
        "intent": intent
    }