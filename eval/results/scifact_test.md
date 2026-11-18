# BEIR/scifact (test)

| Setup | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | ms / query |
|---|---|---|---|---|---|
| dense | 0.717 | 0.685 | 0.761 | 0.842 | 28 |
| bm25 | 0.642 | 0.607 | 0.719 | 0.773 | 54 |
| hybrid | 0.716 | 0.686 | 0.770 | 0.836 | 80 |
| **hybrid+rerank** | 0.726 | 0.696 | 0.779 | 0.850 | 1029 |

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
- dense_top_k: `50`
- sparse_top_k: `50`
- rrf_k: `60`
- fusion_top_k: `50`
