# BEIR/nfcorpus (test)

| Setup | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | ms / query |
|---|---|---|---|---|---|
| dense | 0.348 | 0.536 | 0.129 | 0.164 | 18 |
| bm25 | 0.299 | 0.499 | 0.126 | 0.149 | 20 |
| hybrid | 0.355 | 0.560 | 0.135 | 0.176 | 35 |
| **hybrid+rerank** | 0.359 | 0.578 | 0.138 | 0.167 | 1006 |

## Configuration

- dataset: `nfcorpus`
- split: `test`
- questions: `323`
- documents: `3633`
- chunks: `10521`
- date: `2026-11-18`
- embedding_model: `BAAI/bge-small-en-v1.5`
- reranker_model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- chunk_size: `800`
- chunk_overlap: `120`
- chunk_title_prefix: `True`
- dense_top_k: `50`
- sparse_top_k: `50`
- rrf_k: `60`
- fusion_top_k: `50`
