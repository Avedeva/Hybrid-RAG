# 🎯 Hybrid RAG System

A comprehensive **Retrieval-Augmented Generation (RAG)** system that combines **dense vector retrieval** (semantic similarity) and **BM25 sparse retrieval** (keyword matching) with **Reciprocal Rank Fusion (RRF)** for optimal document retrieval across research papers on tokenizers and self-attention mechanisms.

## 📋 Overview

This project implements a production-ready RAG pipeline designed to answer technical questions about:
- **Tokenizers** - Subword tokenization strategies and implementations
- **Self-Attention Mechanisms** - Transformer architecture fundamentals
- **LLM Internals** - How modern language models process and generate text

The system ingests 6 research papers + the Annotated Transformer tutorial, chunks the content intelligently, and retrieves the most relevant information using a hybrid retrieval approach before passing it to an LLM for answer generation.

## 🏗️ Architecture

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DOCUMENT LOADING & CHUNKING                              │
│    ├─ Docling: PDF extraction & parsing                     │
│    ├─ Recursive Character Splitter: Smart chunking (100 overlap)
│    └─ Chroma Vector DB: Persistence                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. HYBRID RETRIEVAL (Dual Strategy)                         │
│    ├─ Dense Retrieval: Cosine similarity (OpenAI embeddings)│
│    ├─ Sparse Retrieval: BM25 keyword matching               │
│    └─ Fusion: RRF algorithm combines both rankings          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM ANSWER GENERATION                                    │
│    ├─ Context: Top-k fused documents                        │
│    ├─ Model: Nvidia Nemotron-3 Ultra 550B                   │
│    └─ Output: Grounded, concise answers                     │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
Avedeva/
├── PDF's/                      # Research papers & documentation (6 papers)
│   ├── [tokenizer papers]
│   └── [self-attention papers]
├── Chunking.py                 # Document loading & chunking logic
├── Retrieval.py                # Hybrid retrieval + LLM inference (Streamlit app)
├── .gitignore                  # Standard Python ignore patterns
├── .python-version             # Python version specification
├── requirement.txt             # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **API Keys**: 
  - `HF_TOKEN` (HuggingFace - for document processing)
  - `OPENROUTER_FREE_RAG` (OpenRouter - for embeddings & LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Avedeva
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token_here
   OPENROUTER_FREE_RAG=your_openrouter_api_key_here
   ```

5. **Add research papers**
   - Place all PDF files in the `PDF's/` directory
   - The system will automatically discover and load them

### Running the Application

#### Step 1: Index Documents
```bash
python Chunking.py
```

This will:
- 🔁 Load all 6 research papers + Annotated Transformer
- 📃 Extract text using Docling
- ✂️ Split into overlapping chunks (100 token overlap)
- 💾 Store embeddings in Chroma vector database (`my_db/`)

**Output:** `my_db/` directory with indexed vectors

#### Step 2: Launch Interactive RAG Interface
```bash
streamlit run Retrieval.py
```

Navigate to `http://localhost:8501` and start asking questions!

---

## 💡 How It Works

### Chunking Strategy (`Chunking.py`)

**Input:** PDFs + URL (Annotated Transformer)
- Uses **Docling** for robust PDF parsing (handles complex layouts)
- Metadata preservation (source, page numbers)

**Processing:**
- **Splitter:** `RecursiveCharacterTextSplitter`
- **Chunk Size:** Default (optimal for semantic coherence)
- **Overlap:** 100 tokens (reduces context loss at chunk boundaries)
- **Separator:** `\n\n` (respects paragraph structure)

**Output:** Chroma vectorstore with OpenAI text-embedding-3-small embeddings

### Hybrid Retrieval Strategy (`Retrieval.py`)

#### Dense Retrieval (Semantic)
```python
docs_retrieve_similarity = vector_store.similarity_search(Query, k=5)
```
- **Embedder:** OpenAI text-embedding-3-small (via OpenRouter)
- **Metric:** Cosine similarity
- **Returns:** 5 semantically closest documents

#### Sparse Retrieval (Lexical)
```python
retriever_bm25 = BM25Retriever.from_documents(documents=docs)
docs_retrieve_bm25 = retriever_bm25.invoke(Query)
```
- **Algorithm:** BM25 (probabilistic ranking)
- **Strength:** Captures exact keyword matches, acronyms
- **Returns:** 5 lexically relevant documents

#### Reciprocal Rank Fusion (RRF)
```python
fused = rrf([docs_retrieve_bm25, docs_retrieve_similarity], k=60)
```
- **Formula:** Score = Σ 1/(k + rank + 1) across retrievers
- **Benefit:** Combines semantic + keyword strengths, reduces hallucination
- **Returns:** Re-ranked, deduplicated top documents

### LLM Answer Generation

**Prompt:** Context-aware chain-of-thought template
- Forces grounding in retrieved documents
- Prevents fabrication ("say you don't know")
- Temperature: 0.4 (factual, less creative)

**Model:** Nvidia Nemotron-3 Ultra 550B (via OpenRouter)
- Free tier available
- Strong reasoning capabilities
- Optimized for instruction-following

---

## 📊 Expected Performance

### Retrieval Metrics
- **Precision@5:** High (hybrid approach reduces noise)
- **Recall:** Improved vs. single-method retrieval
- **Latency:** ~2-3s per query (embedding + BM25 + LLM)

### Query Examples
✅ "What is byte-pair encoding?"  
✅ "Explain self-attention in transformers"  
✅ "How do position embeddings work?"  
✅ "What are the advantages of multi-head attention?"  

---

## 🔧 Configuration & Customization

### Adjust Retrieval Parameters

**In `Chunking.py`:**
```python
chunker = RecursiveCharacterTextSplitter(
    separators="\n\n",          # Change separator logic
    chunk_overlap=100            # Increase for more context overlap
)
```

**In `Retrieval.py`:**
```python
retriever_bm25.k = 5             # Number of BM25 results
docs_retrieve_similarity = vector_store.similarity_search(Query, k=5)  # Dense results
rrf(..., k=60)                   # RRF reciprocal depth (higher = smoother)
```

### Swap Embedding Model
```python
OpenAIEmbeddings(
    model='openai/text-embedding-3-large',  # Larger model, higher quality
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)
```

### Swap LLM
```python
llm = ChatOpenAI(
    model='openai/gpt-4-turbo',  # Or any OpenRouter model
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.4
)
```

---

## 📚 Research Papers Included

The `PDF's/` directory should contain papers covering:

1. **Tokenization**
   - BPE (Byte-Pair Encoding)
   - WordPiece
   - SentencePiece implementations

2. **Self-Attention & Transformers**
   - Attention Is All You Need
   - Variations & improvements
   - Efficiency optimizations

3. **Supplementary**
   - Annotated Transformer Tutorial (URL-based)

---

## 🛠️ Troubleshooting

### Common Issues

**❌ `ModuleNotFoundError: No module named 'langchain_docling'`**
```bash
pip install langchain-docling
```

**❌ `FAILED due to exception` during PDF loading**
- Check PDF file integrity
- Ensure HF_TOKEN is valid (Docling uses HuggingFace)
- Try removing corrupted PDF and re-running

**❌ Empty or poor retrieval results**
- Check that `my_db/` directory exists (run `Chunking.py` first)
- Verify API key has sufficient quota
- Increase `k` values in retrieval calls

**❌ `OpenRouter` API errors**
- Verify `OPENROUTER_FREE_RAG` key is correct
- Check rate limits (free tier has limits)
- Switch to paid tier if needed

---

## 📈 Future Enhancements

- [ ] **Cross-Encoder Reranking:** Add a cross-encoder model (ms-marco-MiniLM) for final ranking
- [ ] **Query Expansion:** Multi-query and hypothetical document embeddings
- [ ] **Metadata Filtering:** Search by paper title, date, topic
- [ ] **LLM Cache:** Reduce redundant API calls for repeated queries
- [ ] **Evaluation Framework:** Implement RAGAS metrics (context precision, faithfulness)
- [ ] **Batch Inference:** Handle multiple concurrent queries

---

## 📝 License

This project is for educational and research purposes.

---

## 🤝 Contributing

Suggestions for improving retrieval quality or adding new papers? Open an issue or submit a PR!

---

## 📧 Questions?

For issues or feature requests, refer to the project documentation or troubleshooting section above.

---

**Built with ❤️ for understanding LLM internals** 🚀
