"""Debug script to understand why hybrid search misses 2010/45/UE chunks."""
import sys, os, re
import numpy as np
sys.path.insert(0, "src")

from ingestion import BERTEmbedder
from retrieval import FAISSVectorStore
from utils import load_config
import faiss

config = load_config("config.yaml")

emb_cfg = config.get("embeddings", {})
embedder = BERTEmbedder(model_name=emb_cfg.get("model_name"), device="cpu")

vs_cfg = config.get("vector_store", {})
vs = FAISSVectorStore(
    embedding_dim=embedder.get_embedding_dimension(),
    persist_directory=vs_cfg.get("persist_directory", "./data/vector_store"),
)
vs.load()

query = "Quel est l'un des objectifs de la directive 2010/45/UE du 13 juillet 2010 ?"

# 1. Query tokens
tokens_q = vs._tokenize(query)
print("=== QUERY TOKENS ===")
print(tokens_q)

# 2. Chunks containing "2010/45"
print("\n=== CHUNKS CONTAINING '2010/45' ===")
qe = embedder.embed_text(query)
qv = qe.reshape(1, -1).astype("float32")
faiss.normalize_L2(qv)

all_bm25 = vs.bm25.get_scores(tokens_q)

for i, c in enumerate(vs.chunks):
    if "2010/45" in c.get("text", ""):
        tokens_c = vs._tokenize(c["text"])
        has_token = "2010/45/ue" in tokens_c
        vec = vs.index.reconstruct(i).reshape(1, -1)
        sem_score = float(np.dot(qv, vec.T)[0][0])
        bm25_score = all_bm25[i]
        fname = c.get("metadata", {}).get("filename", "?")
        cid = c.get("chunk_id")
        print(f"  idx={i}  chunk_id={cid}  src={fname}")
        print(f"    has token '2010/45/ue': {has_token}")
        print(f"    BM25 score: {bm25_score:.4f}")
        print(f"    Semantic score: {sem_score:.4f}")
        print(f"    text[:150]: {c['text'][:150]}")
        print()

# 3. BM25 top 30
print("=== BM25 TOP 30 ===")
top30_bm25 = np.argsort(all_bm25)[::-1][:30]
for rank, idx in enumerate(top30_bm25):
    c = vs.chunks[idx]
    fname = c.get("metadata", {}).get("filename", "?")
    cid = c.get("chunk_id")
    print(f"  rank={rank+1:2d}  idx={idx:3d}  bm25={all_bm25[idx]:8.4f}  cid={cid}  src={fname}  text[:60]={c['text'][:60]}")

# 4. FAISS top 30
print("\n=== FAISS TOP 30 ===")
scores, indices = vs.index.search(qv, 30)
for rank, (sc, idx) in enumerate(zip(scores[0], indices[0])):
    if idx < 0:
        continue
    c = vs.chunks[idx]
    fname = c.get("metadata", {}).get("filename", "?")
    cid = c.get("chunk_id")
    print(f"  rank={rank+1:2d}  idx={idx:3d}  sem={sc:.4f}  cid={cid}  src={fname}  text[:60]={c['text'][:60]}")

# 5. Hybrid search with top_k=5 (same as API)
print("\n=== HYBRID SEARCH top_k=5, alpha=0.6, candidate_factor=4 (FIXED) ===")
results = vs.hybrid_search(qe, query, top_k=5, alpha=0.6, candidate_factor=4)
for r in results:
    fname = r.get("metadata", {}).get("filename", "?")
    has_2010_45 = "2010/45" in r.get("text", "")
    marker = " <<<< CONTAINS 2010/45/UE" if has_2010_45 else ""
    print(f"  rank={r['rank']}  score={r['score']:.4f}  sem={r.get('semantic_score',0):.4f}  bm25={r.get('bm25_score',0):.4f}  cid={r.get('chunk_id')}  src={fname}{marker}")
    print(f"    text[:100]={r['text'][:100]}")

# 6. Also test with top_k=10
print("\n=== HYBRID SEARCH top_k=10, alpha=0.6, candidate_factor=4 (FIXED) ===")
results2 = vs.hybrid_search(qe, query, top_k=10, alpha=0.6, candidate_factor=4)
for r in results2:
    fname = r.get("metadata", {}).get("filename", "?")
    has_2010_45 = "2010/45" in r.get("text", "")
    marker = " <<<< CONTAINS 2010/45/UE" if has_2010_45 else ""
    print(f"  rank={r['rank']}  score={r['score']:.4f}  sem={r.get('semantic_score',0):.4f}  bm25={r.get('bm25_score',0):.4f}  cid={r.get('chunk_id')}  src={fname}{marker}")
