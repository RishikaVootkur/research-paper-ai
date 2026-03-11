"""
Specialist Agents
-----------------
Three specialist agents, each handling a different type of query:

1. RetrieverAgent:   Factual lookups using RAG pipeline
2. SynthesizerAgent: Multi-paper comparison and synthesis
3. GeneralAgent:     Chitchat and out-of-scope questions

Each agent is a LangGraph node: takes AgentState, returns updated AgentState.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.agents.state import AgentState
from src.agents.llm_provider import get_llm
from src.rag.reranker import RerankedRetriever
from src.rag.query_transform import QueryTransformer
from src.rag.prompts import format_context, format_sources_list, get_prompt, CITATION_INSTRUCTIONS

load_dotenv()


class RetrieverAgent:
    """
    Handles factual questions by retrieving and generating from papers.

    Has the ability to do iterative retrieval: if the first search
    doesn't yield good results, it reformulates the query and tries again.
    """

    def __init__(self, retriever: RerankedRetriever, query_transformer: QueryTransformer):
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.llm = get_llm(temperature=0.1)
        self.output_parser = StrOutputParser()

    def run(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks and generate an answer."""
        question = state["question"]
        question_type = state.get("question_type", "default")
        trace = state.get("agent_trace", [])

        # Decide whether to use HyDE
        use_hyde = self.query_transformer.should_use_hyde(question)
        if use_hyde:
            search_query = self.query_transformer.hyde(question)
        else:
            search_query = question

        # Retrieve
        chunks = self.retriever.search(search_query, top_k=5, fetch_k=20)
        paper_ids = set(c.paper_id for c in chunks)

        # Check if retrieval quality seems low (all chunks from same paper
        # with low scores might mean we should try again)
        if len(chunks) > 0 and chunks[0].score < 2.0 and not use_hyde:
            # Try again with HyDE
            search_query = self.query_transformer.hyde(question)
            chunks_retry = self.retriever.search(search_query, top_k=5, fetch_k=20)
            if chunks_retry and chunks_retry[0].score > chunks[0].score:
                chunks = chunks_retry
                paper_ids = set(c.paper_id for c in chunks)
                use_hyde = True
                trace.append({
                    "agent": "retriever",
                    "action": "iterative_retrieval",
                    "detail": "First retrieval had low scores, retried with HyDE",
                })

        # Format context
        context = format_context(chunks)
        sources = format_sources_list(chunks)

        # Generate answer
        prompt = get_prompt(question_type)
        chain = prompt | self.llm | self.output_parser
        answer = chain.invoke({"context": context, "question": question})

        trace.append({
            "agent": "retriever",
            "action": "retrieve_and_generate",
            "chunks_retrieved": len(chunks),
            "papers_found": len(paper_ids),
            "hyde_used": use_hyde,
        })

        return {
            **state,
            "retrieved_chunks": [{"id": c.chunk_id, "paper": c.paper_title, "score": c.score} for c in chunks],
            "context": context,
            "num_papers": len(paper_ids),
            "search_query": search_query,
            "hyde_used": use_hyde,
            "answer": answer,
            "sources": sources,
            "agent_trace": trace,
        }


class SynthesizerAgent:
    """
    Handles comparison and synthesis questions across multiple papers.

    Uses a wider retrieval (more chunks) and a synthesis-focused prompt
    to combine information from different sources.
    """

    def __init__(self, retriever: RerankedRetriever, query_transformer: QueryTransformer):
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.llm = get_llm(temperature=0.2)
        self.output_parser = StrOutputParser()

    def run(self, state: AgentState) -> AgentState:
        """Retrieve from multiple papers and synthesize."""
        question = state["question"]
        question_type = state.get("question_type", "comparison")
        trace = state.get("agent_trace", [])

        # For synthesis, we want MORE chunks from MORE papers
        # Use expanded query to cast a wider net
        expanded_query = self.query_transformer.expand(question)

        # Retrieve more chunks than usual (top 8 from 30 candidates)
        chunks = self.retriever.search(expanded_query, top_k=8, fetch_k=30)
        paper_ids = set(c.paper_id for c in chunks)

        # If we only got 1 paper, try the original question too
        if len(paper_ids) < 2:
            chunks_original = self.retriever.search(question, top_k=5, fetch_k=20)
            # Merge and deduplicate
            seen_ids = {c.chunk_id for c in chunks}
            for c in chunks_original:
                if c.chunk_id not in seen_ids:
                    chunks.append(c)
                    seen_ids.add(c.chunk_id)
            paper_ids = set(c.paper_id for c in chunks)

        # Format context and generate
        context = format_context(chunks)
        sources = format_sources_list(chunks)

        prompt = get_prompt(question_type)
        chain = prompt | self.llm | self.output_parser
        answer = chain.invoke({"context": context, "question": question})

        trace.append({
            "agent": "synthesizer",
            "action": "synthesize",
            "chunks_retrieved": len(chunks),
            "papers_found": len(paper_ids),
            "expanded_query": expanded_query,
        })

        return {
            **state,
            "retrieved_chunks": [{"id": c.chunk_id, "paper": c.paper_title, "score": c.score} for c in chunks],
            "context": context,
            "num_papers": len(paper_ids),
            "search_query": expanded_query,
            "hyde_used": False,
            "answer": answer,
            "sources": sources,
            "agent_trace": trace,
        }


class GeneralAgent:
    """
    Handles greetings, chitchat, and out-of-scope questions.
    Responds helpfully without using the RAG pipeline.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.5)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant for ML/AI papers. 
The user sent a message that isn't a research question. 
Respond naturally and briefly. If they're greeting you, greet back and 
explain what you can do. If they ask something off-topic, politely 
redirect them to ask about ML/AI research papers.

You can help with:
- Explaining concepts from ML/AI papers (LoRA, attention, RAG, etc.)
- Comparing different methods or approaches
- Summarizing research trends
- Finding information across multiple papers"""),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, state: AgentState) -> AgentState:
        """Respond to general/chitchat messages."""
        question = state["question"]
        trace = state.get("agent_trace", [])

        answer = self.chain.invoke({"question": question})

        trace.append({
            "agent": "general",
            "action": "chitchat_response",
        })

        return {
            **state,
            "answer": answer,
            "sources": [],
            "num_papers": 0,
            "retrieved_chunks": [],
            "context": "",
            "agent_trace": trace,
        }