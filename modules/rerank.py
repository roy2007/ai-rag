# modules/rerank.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BGEReranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
        self.model.eval()

    def rerank(self, query: str, docs: list[dict], top_k=5):
        pairs = [[query, d["text"]] for d in docs]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
        scored = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [{"text": d["text"], "score": s, "source": d["source"]} for d, s in scored[:top_k]]