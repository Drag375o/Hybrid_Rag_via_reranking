# BEIR/scifact (test)

| Setup | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | ms / query |
|---|---|---|---|---|---|
| dense | 0.717 | 0.685 | 0.761 | 0.842 | 20 |
| bm25 | 0.684 | 0.650 | 0.752 | 0.809 | 62 |
| **hybrid** | 0.731 | 0.701 | 0.800 | 0.849 | 54 |
| hybrid+rerank | 0.724 | 0.694 | 0.774 | 0.849 | 955 |

## Configuration

- dataset: `scifact`
- split: `test`
- questions: `300`
- documents: `5183`
- chunks: `14227`
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
