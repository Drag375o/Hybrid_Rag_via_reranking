# BEIR/scifact (test)

| Setup | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | ms / query |
|---|---|---|---|---|---|
| **dense** | 0.713 | 0.683 | 0.767 | 0.833 | 21 |
| bm25 | 0.631 | 0.592 | 0.709 | 0.771 | 37 |
| hybrid | 0.711 | 0.677 | 0.755 | 0.838 | 59 |
| hybrid+rerank | 0.696 | 0.661 | 0.781 | 0.831 | 1030 |

## Configuration

- dataset: `scifact`
- split: `test`
- questions: `300`
- documents: `5183`
- chunks: `13187`
- date: `2026-11-18`
- embedding_model: `BAAI/bge-small-en-v1.5`
- reranker_model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- chunk_size: `800`
- chunk_overlap: `120`
- chunk_title_prefix: `False`
- dense_top_k: `50`
- sparse_top_k: `50`
- rrf_k: `60`
- fusion_top_k: `50`
