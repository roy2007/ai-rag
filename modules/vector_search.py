# modules/vector_search.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

class QdrantVectorStore:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.encoder = SentenceTransformer("./models/bge-large-zh-v1.5")
        self.collection_name = "rag_docs"
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

    def add_texts(self, texts: list[str]):
        vectors = self.encoder.encode(texts).tolist()
        points = [
            PointStruct(id=i, vector=vectors[i], payload={"text": texts[i]})
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k=30):
        query_vec = self.encoder.encode([query]).tolist()[0]
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k
        )
        return [{"text": r.payload["text"], "score": r.score, "source": "vector"} for r in results]