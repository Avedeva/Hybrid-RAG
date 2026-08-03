# Hybrid RAG — Retrieval-Augmented QA over Research Papers

A retrieval-augmented generation pipeline that answers questions over a corpus of PDFs (and web-loaded papers) by combining lexical and semantic retrieval, fusing the results, and reranking before generation — deployed as a Streamlit app.

Built to explore what actually improves retrieval quality in RAG systems beyond naive top-k vector search: hybrid retrieval, rank fusion, and cross-encoder reranking, each added because single-method retrieval leaves clear gaps (semantic search misses exact terms/acronyms, lexical search misses paraphrases).

---

## Why Hybrid, Why Rerank

Vector similarity search alone struggles with:
- **Exact-match terms** — acronyms, symbols, proper nouns, code identifiers that embeddings don't represent well.
- **Ranking quality at the top** — cosine similarity is a decent coarse filter but a weak fine-grained ranker; it can put a tangentially-related chunk above a directly relevant one.

Pure lexical (BM25) retrieval struggles with:
- **Paraphrasing and synonymy** — a question phrased differently from the source text gets no match even when the content is identical.

The pipeline addresses both by retrieving from each independently, fusing the rankings, then applying a cross-encoder — a slower but more accurate model that scores the query and each candidate chunk *jointly* rather than comparing pre-computed embeddings — to re-rank the fused list before it goes to the LLM.

---

## Architecture

```
PDFs + web docs
      │
      ▼
  DoclingLoader (parsing, layout-aware extraction)
      │
      ▼
  RecursiveCharacterTextSplitter (chunking, overlap = 100)
      │
      ▼
  chunks.pkl ──────────────────────────────────────────┐
      │                                                 │
      ▼                                                 ▼
  OpenAI text-embedding-3-small (via OpenRouter)   BM25Retriever
      │                                                 │
      ▼                                                 │
  Chroma vector store (persisted, content-hash          │
  dedup on ingest)                                      │
      │                                                 │
      ▼                                                 ▼
  Dense similarity search (k=10)  ────────────►  Reciprocal Rank Fusion (k=60)
                                                         │
                                                         ▼
                                          cross-encoder/ms-marco-MiniLM-L-6-v2
                                          (rerank fused candidates → top 5)
                                                         │
                                                         ▼
                                          Prompt template + retrieved context
                                                         │
                                                         ▼
                                          nvidia/nemotron-3-ultra-550b-a55b
                                          (via OpenRouter) → answer
                                                         │
                                                         ▼
                                              Streamlit UI (query in, answer out)
```

---

## Pipeline Stages

**1. Ingestion (`load.py`)**
Recursively collects all PDFs from a target directory, adds a supplementary URL source, and loads everything through `DoclingLoader` — chosen for layout-aware parsing (tables, headers, reading order) over naive text extraction. Documents are chunked with `RecursiveCharacterTextSplitter` and serialized to `chunks.pkl` for reuse without re-parsing.

**2. Embedding & storage (`embed.py`)**
Chunks are embedded with OpenAI's `text-embedding-3-small` (served through OpenRouter) and stored in a persisted Chroma collection. Ingestion is idempotent: each chunk is hashed (`md5(source + text)`) and checked against existing IDs in the store, so re-running the script only adds genuinely new chunks instead of duplicating the collection.

**3. Retrieval & generation (`Retrieval.py`)**
For each query:
- **Dense retrieval** — top-10 via Chroma cosine similarity.
- **Lexical retrieval** — top-10 via `BM25Retriever` over the same corpus.
- **Fusion** — Reciprocal Rank Fusion (k=60) merges both ranked lists into one, rewarding chunks that rank well in *either* method without requiring score normalization across methods.
- **Reranking** — a cross-encoder scores each fused candidate jointly against the query and re-sorts; only the top 5 proceed to generation.
- **Generation** — the top-5 chunks are inserted into a prompt template instructing the model to answer only from context (or say it doesn't know), sent to `nvidia/nemotron-3-ultra-550b-a55b` via OpenRouter.
- **UI** — Streamlit handles query input and renders the answer, with spinners marking each pipeline stage.

---

## Stack

| Component | Choice |
|---|---|
| Document parsing | Docling (`langchain_docling`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-3-small` (via OpenRouter) |
| Vector store | Chroma (persisted, local) |
| Lexical retrieval | BM25 (`langchain_community`) |
| Fusion | Reciprocal Rank Fusion, self-implemented |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Nvidia Nemotron (via OpenRouter) |
| Interface | Streamlit |

---

## Repo Structure

```
.
├── PDF's/              # source documents
├── load.py             # parse → chunk → serialize
├── embed.py            # embed → store (idempotent via content hash)
├── Retrieval.py         # hybrid retrieval → fusion → rerank → generate → UI
├── chunks.pkl          # serialized chunks (output of load.py)
├── requirement.txt     # dependencies
└── .gitignore
```

---

## Design Decisions Worth Calling Out

- **Fusion over score-blending**: RRF was chosen over normalizing and summing raw scores from BM25 and cosine similarity, since the two scores live on different, non-comparable scales. RRF only needs each method's *rank order*, sidestepping that problem entirely.
- **Rerank after fusion, not before**: reranking is the most expensive step (cross-encoder inference is quadratic-ish in cost relative to bi-encoder embedding lookup), so it runs once on the fused candidate set rather than on each retrieval method's output separately.
- **Content-hash dedup on ingest**: re-running `embed.py` against an updated PDF folder won't duplicate previously-embedded chunks, which matters for a corpus that grows over time.

---

## Running It

```bash
pip install -r requirement.txt
```

Set environment variables (`.env`):
```
OPENROUTER_FREE_RAG=<your OpenRouter API key>
HF_TOKEN=<your HuggingFace token>
```

```bash
python load.py     # parse PDFs + web source → chunks.pkl
python embed.py    # embed chunks → Chroma store
streamlit run Retrieval.py
```

---

## Known Limitations / Next Steps

- No evaluation harness yet (retrieval precision/recall, answer faithfulness) — next priority for making the "hybrid + rerank improves quality" claim measurable rather than assumed.
- Single fixed `k` at each stage (10/10/60/5) — not yet tuned against a labeled query set.
- No chunk-size/overlap ablation — current values (separator-based, overlap=100) are a reasonable default, not a benchmarked choice.
- Generation model is a free-tier OpenRouter model, chosen for cost during development rather than benchmarked quality.
