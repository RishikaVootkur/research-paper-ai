# Research Paper Intelligence Platform

An AI-powered system for intelligent research paper analysis and discovery, built with RAG (Retrieval-Augmented Generation), multi-agent architecture, and fine-tuned ML models.

## What It Does

Ask questions about ML/AI research papers in plain English and get cited, grounded answers drawn from actual papers. The system intelligently routes your questions to specialized agents, retrieves relevant content using hybrid search, and generates well-cited answers.

## Architecture

```
User Question
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator (LangGraph)             │
│                                                              │
│  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │  Router   │────►│  Specialist   │────►│     Critic      │  │
│  │  Agent    │     │  Agent        │     │     Agent       │  │
│  └──────────┘     └──────────────┘     └─────────────────┘  │
│       │            ├─ Retriever                    │          │
│       │            ├─ Synthesizer          Revision loop     │
│       │            └─ General              if score < 3      │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                               │
│                                                              │
│  Query ──► [HyDE] ──► Hybrid Search ──► Re-rank ──► LLM     │
│                       (Vector+BM25)   (Cross-Encoder)        │
│                            │                                  │
│                      ┌─────┴─────┐                           │
│                      │ ChromaDB  │ 3,476 chunks              │
│                      │ 35 papers │                            │
│                      └───────────┘                           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    ML Layer                                    │
│                                                              │
│  ┌─────────────────┐     ┌──────────────────────┐            │
│  │ Topic Classifier │     │ Paper Recommender     │            │
│  │ (DistilBERT      │     │ (Embedding Similarity) │            │
│  │  fine-tuned)     │     │                        │            │
│  └─────────────────┘     └──────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent Orchestration | LangGraph | State machine for multi-agent coordination |
| LLM | Groq (Llama 3.3 70B) | Free API, fast inference |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Free, local, 384-dim |
| Vector Database | ChromaDB | Free, local, persistent |
| Keyword Search | BM25 (rank-bm25) | Complements vector search |
| Re-ranking | Cross-Encoder (ms-marco-MiniLM) | Precision on top candidates |
| Topic Classifier | DistilBERT (fine-tuned) | Paper categorization |
| Paper Source | ArXiv API | Free, 2M+ papers |
| PDF Processing | PyMuPDF | Fast text extraction |

## Project Structure

```
research-paper-ai/
├── src/
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py       # Fetches papers from ArXiv
│   │   ├── pdf_processor.py       # Extracts text, creates chunks
│   │   ├── vector_store.py        # Embeds and stores in ChromaDB
│   │   └── pipeline.py            # End-to-end ingestion + CLI
│   ├── rag/
│   │   ├── hybrid_retriever.py    # Vector + BM25 hybrid search
│   │   ├── reranker.py            # Cross-encoder re-ranking
│   │   ├── query_transform.py     # HyDE query transformation
│   │   ├── prompts.py             # Question-type-aware prompts
│   │   ├── rag_chain.py           # Core RAG pipeline
│   │   └── conversation.py        # Multi-turn conversation memory
│   ├── agents/
│   │   ├── state.py               # Shared agent state definition
│   │   ├── router.py              # Routes to specialist agents
│   │   ├── specialists.py         # Retriever, Synthesizer, General
│   │   ├── critic.py              # Quality review agent
│   │   └── graph.py               # LangGraph orchestration
│   └── ml/
│       ├── prepare_classification_data.py  # ArXiv dataset builder
│       ├── topic_classifier.py    # DistilBERT fine-tuning
│       ├── classify_papers.py     # Classify ingested papers
│       └── recommender.py         # Paper similarity recommendations
├── tests/
│   ├── test_rag_pipeline.py       # RAG pipeline tests (4/5 passing)
│   └── test_agents.py             # Agent system tests (5/5 passing)
├── models/
│   └── topic_classifier/          # Fine-tuned DistilBERT model
├── data/
│   ├── processed/                 # Metadata, chunks, topics
│   └── ml/                        # Classification training data
├── demo.py                        # Interactive terminal demo
└── chroma_db/                     # Persistent vector database
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/research-paper-ai.git
cd research-paper-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys (Groq, Google AI Studio, HuggingFace)
```

## Usage

```bash
# Interactive demo (multi-agent system)
python demo.py

# Ingest papers
python src/ingestion/pipeline.py --query "retrieval augmented generation" -n 10
python src/ingestion/pipeline.py --ids 2005.11401 1706.03762
python src/ingestion/pipeline.py --status

# Run tests
python tests/test_rag_pipeline.py
python tests/test_agents.py

# Classify papers by topic
python src/ml/classify_papers.py

# Get paper recommendations
python src/ml/recommender.py
```

## ML Experiments

### Topic Classification (Fine-tuned DistilBERT)

**Task:** Classify research papers into 5 categories (NLP, Computer Vision, Machine Learning, Artificial Intelligence, Information Retrieval) based on title + abstract.

**Dataset:** 500 papers from ArXiv (100 per category), split 70/15/15.

**Model:** `distilbert-base-uncased` fine-tuned for 10 epochs on Apple Silicon (MPS).

| Metric | Score |
|--------|-------|
| Validation Accuracy | 57.3% |
| Test Accuracy | 50.7% |
| Best Per-Class F1 | 74% (Information Retrieval) |

**Per-class results:**
| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| NLP | 0.47 | 0.53 | 0.50 |
| Computer Vision | 0.67 | 0.56 | 0.61 |
| Machine Learning | 0.50 | 0.50 | 0.50 |
| Artificial Intelligence | 0.24 | 0.24 | 0.24 |
| Information Retrieval | 0.71 | 0.77 | 0.74 |

**Analysis:** Categories with distinctive vocabulary (CV, IR) perform well. AI has lowest F1 because it overlaps heavily with ML and NLP — most AI papers could arguably fit multiple categories. Validation accuracy peaked at epoch 2 (57.3%) then declined, indicating overfitting on our small dataset. More training data would improve performance.

### Paper Recommendations (Embedding Similarity)

**Method:** Content-based filtering using averaged chunk embeddings per paper. Cosine similarity for ranking.

**Results:** Recommendations are semantically coherent — LoRA's top recommendations are fine-tuning papers, Attention's top recommendations are transformer-related papers, and text-based queries about RAG evaluation correctly surface RAG survey papers.

## Current Status

- [x] **Week 1:** Data ingestion pipeline (35 papers, 3,476 chunks)
- [x] **Week 2:** Advanced RAG (hybrid search, re-ranking, HyDE, conversation memory)
- [x] **Week 3:** Multi-agent system (Router → Specialist → Critic via LangGraph)
- [x] **Week 4:** ML layer (fine-tuned topic classifier, paper recommendations)
- [ ] Week 5: API + Frontend + Evaluation
- [ ] Week 6: Docker, deployment, portfolio polish