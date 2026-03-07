"""
Streamlit Frontend
------------------
Web UI for the Research Paper Intelligence Platform.

Run: streamlit run frontend/app.py
"""

import os
import sys
import streamlit as st
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Research Paper AI",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# Load models (cached so they only load once)
# ============================================================
@st.cache_resource
def load_orchestrator():
    """Load the multi-agent orchestrator (cached across reruns)."""
    from src.agents.graph import AgentOrchestrator
    return AgentOrchestrator()


@st.cache_resource
def load_recommender():
    """Load the paper recommender."""
    from src.ml.recommender import PaperRecommender
    return PaperRecommender()


@st.cache_resource
def load_classifier():
    """Load the topic classifier."""
    model_path = "models/topic_classifier"
    if os.path.exists(model_path):
        from src.ml.topic_classifier import TopicClassifier
        return TopicClassifier.load(model_path)
    return None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("📚 Research Paper AI")
    st.markdown("*Multi-agent RAG system for ML/AI papers*")

    st.divider()

    page = st.radio(
        "Navigation",
        ["💬 Ask Questions", "🔍 Explore Papers", "🏷️ Classify", "ℹ️ About"],
    )

    st.divider()

    # Show database stats
    try:
        orchestrator = load_orchestrator()
        stats = orchestrator.retriever_agent.retriever.hybrid_retriever.vector_store.get_collection_stats()
        st.metric("Papers Indexed", stats["papers"])
        st.metric("Total Chunks", stats["total_chunks"])
    except Exception:
        st.warning("Loading models...")


# ============================================================
# Page: Ask Questions
# ============================================================
if page == "💬 Ask Questions":
    st.header("💬 Ask Questions About Research Papers")
    st.markdown("Ask anything about ML/AI research. Your question is routed to specialized agents for the best answer.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                meta = message["metadata"]
                cols = st.columns(4)
                cols[0].caption(f"🔀 {meta.get('route', 'N/A')}")
                cols[1].caption(f"📋 {meta.get('question_type', 'N/A')}")
                cols[2].caption(f"⭐ {meta.get('quality_score', 'N/A')}/5")
                cols[3].caption(f"📚 {meta.get('num_papers', 0)} papers")

                if meta.get("sources"):
                    with st.expander("📎 Sources"):
                        for src in meta["sources"]:
                            st.markdown(f"- **{src['title'][:60]}...** (p.{src['page']})")

    # Chat input
    if prompt := st.chat_input("Ask about ML/AI research papers..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    orchestrator = load_orchestrator()
                    start = time.time()
                    result = orchestrator.run(prompt)
                    elapsed = time.time() - start

                    st.markdown(result["answer"])

                    # Metadata
                    cols = st.columns(4)
                    cols[0].caption(f"🔀 {result['route']}")
                    cols[1].caption(f"📋 {result['question_type']}")
                    cols[2].caption(f"⭐ {result['quality_score']}/5")
                    cols[3].caption(f"⏱️ {elapsed:.1f}s")

                    if result["sources"]:
                        with st.expander("📎 Sources"):
                            for src in result["sources"]:
                                st.markdown(f"- **{src['title'][:60]}...** (p.{src['page']})")

                    with st.expander("🔗 Agent Trace"):
                        for step in result["agent_trace"]:
                            agent = step.get("agent", "unknown")
                            action = step.get("action", "unknown")
                            st.text(f"  → [{agent}] {action}")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "route": result["route"],
                            "question_type": result["question_type"],
                            "quality_score": result["quality_score"],
                            "num_papers": result["num_papers"],
                            "sources": result["sources"],
                        },
                    })

                except Exception as e:
                    st.error(f"Error: {e}")

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# Page: Explore Papers
# ============================================================
elif page == "🔍 Explore Papers":
    st.header("🔍 Explore Papers & Recommendations")

    tab1, tab2 = st.tabs(["📚 Paper List", "🔮 Recommendations"])

    with tab1:
        try:
            orchestrator = load_orchestrator()
            store = orchestrator.retriever_agent.retriever.hybrid_retriever.vector_store
            stats = store.get_collection_stats()

            st.subheader(f"Indexed Papers ({stats['papers']})")

            # Load ingestion log for titles
            import json
            log_path = "data/processed/ingestion_log.json"
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log = json.load(f)

                for pid in stats.get("paper_ids", []):
                    info = log.get("ingested_papers", {}).get(pid, {})
                    title = info.get("title", "Unknown")
                    chunks = info.get("chunks", "?")
                    st.markdown(f"**[{pid}]** {title} — *{chunks} chunks*")
            else:
                for pid in stats.get("paper_ids", []):
                    st.markdown(f"**{pid}**")

        except Exception as e:
            st.error(f"Error loading papers: {e}")

    with tab2:
        st.subheader("Find Similar Papers")

        rec_method = st.radio("Search by:", ["Text Description", "Paper ID"])

        if rec_method == "Text Description":
            text_input = st.text_area("Describe what you're looking for:", height=100)
            top_k = st.slider("Number of recommendations:", 1, 10, 5)

            if st.button("Find Papers") and text_input:
                with st.spinner("Finding similar papers..."):
                    recommender = load_recommender()
                    results = recommender.recommend_by_text(text_input, top_k=top_k)

                    for i, r in enumerate(results, 1):
                        similarity_pct = r["similarity"] * 100
                        st.markdown(f"**{i}. {r['title'][:70]}...**")
                        st.caption(f"Similarity: {similarity_pct:.1f}% | Authors: {r['authors'][:50]}")
                        st.divider()

        else:
            paper_id = st.text_input("Enter ArXiv Paper ID:", placeholder="e.g., 2106.09685v2")
            top_k = st.slider("Number of recommendations:", 1, 10, 5, key="rec_k")

            if st.button("Find Similar") and paper_id:
                with st.spinner("Finding similar papers..."):
                    recommender = load_recommender()
                    results = recommender.recommend_by_id(paper_id, top_k=top_k)

                    if results:
                        for i, r in enumerate(results, 1):
                            similarity_pct = r["similarity"] * 100
                            st.markdown(f"**{i}. {r['title'][:70]}...**")
                            st.caption(f"Similarity: {similarity_pct:.1f}%")
                            st.divider()
                    else:
                        st.warning("Paper not found in database.")


# ============================================================
# Page: Classify
# ============================================================
elif page == "🏷️ Classify":
    st.header("🏷️ Paper Topic Classification")
    st.markdown("Classify a paper into: NLP, Computer Vision, Machine Learning, AI, or Information Retrieval")

    text_input = st.text_area(
        "Paste paper title + abstract:",
        height=150,
        placeholder="Attention Is All You Need. We propose a new network architecture...",
    )

    if st.button("Classify") and text_input:
        classifier = load_classifier()
        if classifier:
            result = classifier.predict(text_input)

            st.success(f"**Predicted Topic: {result['label']}** ({result['confidence']:.1%} confidence)")

            st.subheader("All Scores")
            import pandas as pd
            scores_df = pd.DataFrame(
                list(result["all_scores"].items()),
                columns=["Topic", "Score"]
            ).sort_values("Score", ascending=False)
            st.bar_chart(scores_df.set_index("Topic"))
        else:
            st.error("Topic classifier not loaded. Train it first with `python src/ml/topic_classifier.py`")


# ============================================================
# Page: About
# ============================================================
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### Research Paper Intelligence Platform

    An AI-powered system for intelligent research paper analysis and discovery.

    **Architecture:**
    - **Multi-Agent System** (LangGraph): Router, Retriever, Synthesizer, General, Critic
    - **Advanced RAG**: Hybrid search (Vector + BM25) → Cross-encoder re-ranking → HyDE
    - **ML Models**: Fine-tuned DistilBERT topic classifier, embedding-based recommender
    - **Backend**: FastAPI with auto-generated docs
    - **Frontend**: Streamlit (this UI)

    **Tech Stack:**
    LangGraph, LangChain, Groq (Llama 3.3 70B), ChromaDB, Sentence Transformers,
    PyTorch, FastAPI, Streamlit, HuggingFace Transformers

    **Data:**
    35 ML/AI research papers (3,476 chunks) indexed from ArXiv.

    Built as a portfolio project demonstrating production-grade ML/GenAI engineering.
    """)