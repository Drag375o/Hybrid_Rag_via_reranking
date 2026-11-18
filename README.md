# Hybrid RAG via Reranking

A fully local, free-to-run **Retrieval-Augmented Generation (RAG)** system that combines **dense (semantic) search** and **sparse (BM25 keyword) search**, fuses the results with **Reciprocal Rank Fusion**, refines them with a **cross-encoder reranker**, and generates grounded, cited answers with a **local LLM via Ollama**.

It comes with a **Streamlit UI** that shows each step of the pipeline, and an **evaluation harness** that measures how much each stage contributes.

> **Status:** 🚧 Work in progress. See the [Roadmap](#roadmap) for what is implemented so far.

---

## Why hybrid + reranking?

| Approach | Good at | Weak at |
|---|---|---|
| **Dense search** (embeddings) | Matching *meaning*: "fix a flat tire" ≈ "repair a punctured wheel" | Exact terms, IDs, codes, rare names |
| **Sparse search** (BM25) | Exact keywords: `Error 0x80070005`, `Section 4.2` | Synonyms and paraphrases |
| **Hybrid (RRF)** | Combines the strengths of both | Its ranking is still approximate |
| **+ Cross-encoder reranker** | Reads the query and each passage together for precise relevance scoring | Too slow to run on the whole corpus, so it only scores the shortlist |

Retrieval stays fast because the cheap retrievers cast a wide net (~50 candidates), and answers stay accurate because the expensive reranker only scores that shortlist to pick the best ~5.

---

## Architecture

```mermaid
flowchart TD
    A[Documents<br/>PDF / TXT / MD / DOCX] --> B[Loader & Parser]
    B --> C[Chunker<br/>with metadata]
    C --> D[(Dense index<br/>ChromaDB + bge embeddings)]
    C --> E[(Sparse index<br/>BM25)]

    Q[User question] --> D
    Q --> E
    D --> F[Hybrid fusion<br/>Reciprocal Rank Fusion<br/>top ~50]
    E --> F
    F --> G[Cross-encoder reranker<br/>top ~5]
    G --> H[Local LLM via Ollama<br/>answer + citations]
    H --> UI[Streamlit UI]
```

### Pipeline stages

1. **Ingestion**: Load documents, split them into overlapping chunks, and attach metadata (source file, page, chunk id).
2. **Indexing**: Embed chunks into a ChromaDB vector store and build a BM25 keyword index over the same chunks.
3. **Hybrid retrieval**: Query both indexes and merge the results with Reciprocal Rank Fusion:

   $$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

4. **Reranking**: Score each (question, chunk) pair with a cross-encoder and keep the top results.
5. **Generation**: Prompt the local LLM to answer **only** from the reranked context and cite its sources.
   - **Relevance gate**: if no passage scores above `min_relevance` with the reranker, the system answers "not found" without calling the LLM, so small models can't answer from their own knowledge.
   - **Citation check**: each answer sentence is compared with the passage it cites. Citations whose passage shares too few of the sentence's words are flagged, together with the best-matching passage. Small models like the default `qwen2.5:1.5b` sometimes cite the wrong passage or forget to cite; larger models cite more reliably.

---

## Tech stack (100% free & local)

| Component | Choice | Role |
|---|---|---|
| Embeddings | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via `sentence-transformers` | Turns text into meaning vectors |
| Vector store | [ChromaDB](https://www.trychroma.com/) (persistent, on disk) | Stores and searches embeddings |
| Keyword search | [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) | Classic BM25 lexical retrieval |
| Fusion | Reciprocal Rank Fusion | Merges the dense and sparse rankings |
| Reranker | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (swap in [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) for higher accuracy) | Precise relevance scoring |
| LLM | [Ollama](https://ollama.com/) (default `qwen2.5:1.5b`; `qwen2.5:3b` or `qwen2.5:7b` give better answers if your hardware allows) | Writes the final answer |
| UI | [Streamlit](https://streamlit.io/) | Chat interface with pipeline inspection |
| Evaluation | [BEIR](https://github.com/beir-cellar/beir) benchmark (default: SciFact) | Measures retrieval quality |

---

## Project structure

```
Hybrid_Rag_via_reranking/
├── src/hybrid_rag/
│   ├── config.py            # Central settings (models, top-k, paths)
│   ├── schema.py            # Shared data types (Document, Chunk, ScoredChunk, Answer)
│   ├── ingestion/           # Document loaders and chunking
│   ├── indexing/            # Dense (ChromaDB) and sparse (BM25) indexes
│   ├── retrieval/           # Hybrid retrieval + RRF fusion
│   ├── reranking/           # Cross-encoder reranker
│   ├── generation/          # Ollama client, prompts and citation check
│   ├── pipeline.py          # End-to-end RAG pipeline
│   └── ui_helpers.py        # Formatting helpers for the UI
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── eval/                    # Benchmark loading, metrics, experiment runner (planned)
├── scripts/                 # CLI helpers (e.g. ingest documents)
├── tests/                   # Unit tests and Streamlit app tests
├── data/                    # Your documents and indexes (git-ignored)
├── requirements.txt
└── README.md
```

---

## Getting started

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running
- About 8 GB of RAM (16 GB recommended) and about 2 GB of free disk space for models

### Installation

```bash
# 1. Clone the repository
git clone git@github.com:Drag375o/Hybrid_Rag_via_reranking.git
cd Hybrid_Rag_via_reranking

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local LLM
ollama pull qwen2.5:1.5b

# 5. (Optional) Override default settings
cp .env.example .env

# 6. Check that everything is installed correctly
pytest
```

The embedding and reranker models download automatically from Hugging Face on first run. Every setting (models, chunk size, top-k values) is defined in `src/hybrid_rag/config.py` and can be overridden with a `RAG_`-prefixed environment variable or in `.env`.

### Usage

#### Chat UI

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. In the sidebar, upload PDF/TXT/MD/DOCX files (or drop them in `data/docs/`) and click **Save & rebuild index**. For every answer, the UI shows:

- the answer, streamed as it is written, with **[n]** citations
- **citation-check warnings** when a sentence cites a passage that doesn't support it, or has no citation
- a note when the **relevance gate** stopped an off-topic question before it reached the LLM
- **Sources**: the passages the LLM was given, with relevance scores
- **How these passages were chosen**: every candidate's rank in dense search, BM25, fusion and reranking

The sidebar switches between hybrid, dense-only and BM25-only retrieval, turns reranking on or off, and sets how many passages the LLM gets.

#### Command line

```bash
# Preview how your documents are loaded and chunked (put them in data/docs/ first)
python -m hybrid_rag.ingestion --show 3

# Build the dense (ChromaDB) and BM25 indexes from data/docs
# (re-run after adding or changing documents; it rebuilds from scratch)
python -m scripts.ingest

# Search from the terminal and see each result's rank in the dense and BM25 lists
python -m hybrid_rag.retrieval "your question" --mode hybrid   # or: dense | sparse

# Add the cross-encoder reranker and see how far each chunk moved
python -m hybrid_rag.retrieval "your question" --rerank

# Ask a question and get a cited answer from the local LLM
python -m hybrid_rag "your question"
```

> Coming soon (see the roadmap):

```bash
# Run the retrieval benchmark
python -m eval.run_eval --dataset scifact
```

---

## Evaluation

The evaluation harness compares four retrieval setups on the same benchmark so the contribution of each stage is measurable:

| Setup | Description |
|---|---|
| Dense only | Embedding search |
| BM25 only | Keyword search |
| Hybrid | Dense + BM25 fused with RRF |
| **Hybrid + Rerank** | Hybrid, then the cross-encoder reranker |

**Metrics:** Recall@k, MRR@10, and nDCG@10.

Results will be published here once the harness is complete.

---

## Roadmap

- [x] Project plan and README
- [x] Project scaffold (config, dependencies, `.gitignore`)
- [x] Document ingestion and chunking
- [x] Dense index (ChromaDB + bge embeddings)
- [x] Sparse index (BM25)
- [x] Hybrid retrieval with Reciprocal Rank Fusion
- [x] Cross-encoder reranking
- [x] Answer generation with Ollama and citations
- [x] Streamlit UI with pipeline inspection
- [ ] Evaluation harness on a BEIR benchmark
- [ ] Results and analysis
