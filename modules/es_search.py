# modules/es_search.py
from elasticsearch import Elasticsearch
import jieba

class ESSearcher:
    def __init__(self):
        self.es = Elasticsearch("http://localhost:19200")
        self.index_name = "rag_docs"
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body={
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "ik_max_word"},
                        "raw_text": {"type": "keyword"}
                    }
                }
            })

    def add_texts(self, texts: list[str]):
        actions = [{"_index": self.index_name, "_source": {"content": t, "raw_text": t}} for t in texts]
        from elasticsearch.helpers import bulk
        bulk(self.es, actions)

    def search(self, query: str, top_k=20):
        # 使用中文分词（需提前安装 IK 分词器？→ 改用 simple 分析器避免依赖）
        resp = self.es.search(
            index=self.index_name,
            query={"match": {"content": query}},
            size=top_k
        )
        return [{"text": hit["_source"]["raw_text"], "score": hit["_score"], "source": "es"} 
                for hit in resp["hits"]["hits"]]