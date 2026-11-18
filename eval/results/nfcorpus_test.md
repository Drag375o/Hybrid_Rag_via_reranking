# BEIR/nfcorpus (test)

| Setup | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | ms / query |
|---|---|---|---|---|---|
| dense | 0.348 | 0.536 | 0.129 | 0.164 | 20 |
| bm25 | 0.323 | 0.527 | 0.132 | 0.157 | 11 |
| **hybrid** | 0.366 | 0.571 | 0.137 | 0.178 | 32 |
| hybrid+rerank | 0.362 | 0.579 | 0.138 | 0.169 | 845 |

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
- bm25_stemming: `True`
- dense_top_k: `50`
- sparse_top_k: `50`
- rrf_k: `60`
- fusion_top_k: `50`
