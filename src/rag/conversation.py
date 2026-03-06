"""
Conversation Manager
--------------------
Adds multi-turn conversation support to the RAG chain.

Uses query rewriting: takes a follow-up question + chat history,
rewrites it into a standalone question, then runs the normal
RAG pipeline. This is more efficient than stuffing all history
into the prompt.

Usage:
    from src.rag.conversation import ConversationalRAG
    chat = ConversationalRAG()
    r1 = chat.chat("How does LoRA work?")
    r2 = chat.chat("What are its limitations?")  # understands "its" = LoRA
    chat.reset()  # clear history for a new topic
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.rag.rag_chain import RAGChain

load_dotenv()


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter. Given a conversation history and a follow-up question, 
rewrite the follow-up question into a standalone question that captures the full context.

RULES:
- The rewritten question must be self-contained (understandable without the history).
- Resolve all pronouns (it, they, this, that, its, their) using the history.
- Keep the rewritten question concise and natural.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question, nothing else."""),
    ("human", """CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION: {question}

Rewritten standalone question:"""),
])


class ConversationalRAG:
    """
    RAG chain with conversation memory.

    Keeps track of Q&A history and rewrites follow-up questions
    into standalone queries before retrieval.
    """

    def __init__(self, **rag_kwargs):
        """
        Args:
            **rag_kwargs: Arguments passed to RAGChain
        """
        self.rag = RAGChain(**rag_kwargs)

        # Lightweight LLM for query rewriting (same model, higher temp not needed)
        self.rewrite_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=200,
        )
        self.rewrite_chain = REWRITE_PROMPT | self.rewrite_llm | StrOutputParser()

        # Conversation history: list of (question, answer) tuples
        self.history: list[tuple[str, str]] = []

        # How many past turns to keep (prevents history from growing forever)
        self.max_history = 5

        print("Conversational RAG initialized.")

    def _format_history(self) -> str:
        """Format conversation history for the rewrite prompt."""
        if not self.history:
            return "No previous conversation."

        parts = []
        for q, a in self.history[-self.max_history:]:
            # Only include a summary of the answer (first 200 chars)
            # to keep the rewrite prompt small
            short_answer = a[:200] + "..." if len(a) > 200 else a
            parts.append(f"User: {q}\nAssistant: {short_answer}")

        return "\n\n".join(parts)

    def _rewrite_query(self, question: str) -> str:
        """
        Rewrite a follow-up question into a standalone question.
        Only rewrites if there's conversation history.
        """
        if not self.history:
            return question

        history_str = self._format_history()

        rewritten = self.rewrite_chain.invoke({
            "history": history_str,
            "question": question,
        })

        return rewritten.strip()

    def chat(self, question: str, top_k: int = None) -> dict:
        """
        Send a message in the conversation.

        If there's history, the question is rewritten to be standalone
        before being sent to the RAG pipeline.

        Args:
            question: User's message
            top_k: Override number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and conversation metadata
        """
        # Step 1: Rewrite if needed
        original_question = question
        if self.history:
            print(f"\n💬 Original: {question}")
            question = self._rewrite_query(question)
            print(f"✏️  Rewritten: {question}")
        
        # Step 2: Run RAG pipeline with the (possibly rewritten) question
        result = self.rag.query(question, top_k=top_k)

        # Step 3: Store in history
        self.history.append((original_question, result["answer"]))

        # Step 4: Add conversation metadata to result
        result["original_question"] = original_question
        result["rewritten_question"] = question
        result["turn_number"] = len(self.history)

        return result

    def reset(self):
        """Clear conversation history. Use when switching topics."""
        self.history = []
        print("🔄 Conversation history cleared.")

    def get_history(self) -> list[tuple[str, str]]:
        """Get the current conversation history."""
        return self.history.copy()


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    chat = ConversationalRAG()

    # Simulate a multi-turn conversation
    conversations = [
        # Turn 1: Standalone question
        "What is LoRA and how does it work?",
        # Turn 2: Follow-up with pronoun "it"
        "What are its main advantages over full fine-tuning?",
        # Turn 3: Follow-up with "this method"
        "Has this method been applied in federated learning settings?",
    ]

    for question in conversations:
        print(f"\n{'='*60}")
        print(f"USER: {question}")
        print(f"{'='*60}")

        result = chat.chat(question)

        if result["original_question"] != result["rewritten_question"]:
            print(f"\n✏️  Query rewritten to: {result['rewritten_question']}")

        print(f"\n📝 Type: {result['question_type']} | Turn: {result['turn_number']}")
        print(f"\nANSWER:\n{result['answer'][:500]}...")
        print(f"\nSOURCES ({result['num_papers']} papers):")
        for src in result["sources"]:
            print(f"  - {src['title'][:55]}...")