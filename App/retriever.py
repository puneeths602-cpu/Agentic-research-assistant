"""Vector store (ChromaDB) with a local hashing embedder: works offline."""
import glob
import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

_vec = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2", stop_words="english")


class HashEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return _vec.transform(input).toarray().tolist()

    @staticmethod
    def name() -> str:
        return "hash-embedding"


def chunk(text: str, size: int = 280, overlap: int = 60):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


class Retriever:
    def __init__(self, docs_dir="data/docs"):
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("knowledge_base")
        except Exception:
            pass
        self.col = client.create_collection("knowledge_base", embedding_function=HashEmbedding(),
                                            metadata={"hnsw:space": "cosine"})
        ids, docs, metas = [], [], []
        for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
            src = os.path.basename(path)
            for n, c in enumerate(chunk(open(path).read())):
                ids.append(f"{src}-{n}"); docs.append(c); metas.append({"source": src})
        self.col.add(ids=ids, documents=docs, metadatas=metas)

    def search(self, query: str, k: int = 3):
        r = self.col.query(query_texts=[query], n_results=k)
        return [{"text": d, "source": m["source"]} for d, m in zip(r["documents"][0], r["metadatas"][0])]
