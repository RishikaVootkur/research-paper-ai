# Research Paper Intelligence Platform

An AI-powered system for intelligent research paper analysis and discovery, built with RAG (Retrieval-Augmented Generation) and multi-agent architecture.

## What It Does

Ask questions about ML/AI research papers in plain English and get cited, grounded answers drawn from actual papers. The system fetches papers from ArXiv, processes them, and makes them searchable using semantic similarity.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           Ingestion Pipeline             │
                    │                                          │
  ArXiv API ──────► │  Fetch ──► Extract ──► Chunk ──► Embed   │
                    │                                          │
                    └──────────────────┬───────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    ChromaDB     │
                              │  Vector Store   │
                              │  (3476 chunks,  │
                              │   35 papers)    │
                              └────────┬────────┘
                                       │
                                       ▼
                              Semantic Search
                           (query by meaning,
                            not just keywords)
```

**Coming Soon:** RAG pipeline, Multi-agent system, Fine-tuned models, Web UI

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Paper Source | ArXiv API | Free, open, 2M+ papers |
| PDF Extraction | PyMuPDF | Fast, handles complex layouts |
| Text Chunking | LangChain RecursiveCharacterTextSplitter | Preserves paragraph structure |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Free, local, 384-dim vectors |
| Vector Database | ChromaDB | Free, local, persistent storage |
| LLM | Groq (Llama 3.3 70B) | Free API, fast inference |

## Project Structure

```
research-paper-ai/
├── src/
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py    # Fetches papers from ArXiv
│   │   ├── pdf_processor.py    # Extracts text, creates chunks
│   │   ├── vector_store.py     # Embeds and stores in ChromaDB
│   │   └── pipeline.py         # End-to-end orchestration + CLI
│   ├── rag/                    # (Week 2) RAG pipeline
│   ├── agents/                 # (Week 3) Multi-agent system
│   ├── ml/                     # (Week 4) Fine-tuning & classification
│   └── api/                    # (Week 5) FastAPI backend
├── data/
│   └── processed/              # Metadata and chunk JSONs
├── chroma_db/                  # Persistent vector database
├── frontend/                   # (Week 5) Streamlit UI
├── tests/
└── configs/
```

## Setup

```bash
# Clone the repo
git clone https://github.com/RishikaVootkur/research-paper-ai.git
cd research-paper-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env with your keys (Groq, Google AI Studio, HuggingFace)
```

## Usage

```bash
# Ingest papers by search query
python src/ingestion/pipeline.py --query "retrieval augmented generation" -n 10

# Ingest specific papers by ArXiv ID
python src/ingestion/pipeline.py --ids 2005.11401 1706.03762

# Ingest by ArXiv category
python src/ingestion/pipeline.py --category cs.CL -n 5

# Search the database
python src/ingestion/pipeline.py --search "How does attention work in transformers?"

# Check database status
python src/ingestion/pipeline.py --status

# Reset database
python src/ingestion/pipeline.py --reset
```

## Current Status

- [x] **Week 1:** Data ingestion pipeline (fetch, extract, chunk, embed, store)
- [ ] Week 2: RAG pipeline (hybrid search, re-ranking, LLM-powered answers)
- [ ] Week 3: Multi-agent system (Router, Retriever, Synthesizer, Critic)
- [ ] Week 4: ML layer (LoRA fine-tuning, topic classification)
- [ ] Week 5: API + Frontend + Evaluation
- [ ] Week 6: Docker, deployment, portfolio polish