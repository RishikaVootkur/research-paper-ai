"""
Agent Tracing & Logging
-----------------------
Logs every agent run with full trace data for debugging
and observability. Saves to JSON files for analysis.

This replaces the need for external tools like LangSmith
while providing the same visibility into agent decisions.
"""

import os
import json
from datetime import datetime


class AgentTracer:
    """
    Logs agent runs with full trace data.

    Each run is saved as a JSON entry with:
    - Question, answer, sources
    - Agent trace (which agents ran, what they did)
    - Quality score, timing
    - Retrieval details (chunks, papers, HyDE usage)
    """

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "agent_traces.jsonl")

    def log_run(self, result: dict, elapsed_seconds: float = None):
        """
        Log a single agent run.

        Args:
            result: Output from AgentOrchestrator.run()
            elapsed_seconds: How long the run took
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": result.get("question", ""),
            "route": result.get("route", ""),
            "question_type": result.get("question_type", ""),
            "quality_score": result.get("quality_score", 0),
            "num_papers": result.get("num_papers", 0),
            "hyde_used": result.get("hyde_used", False),
            "answer_length": len(result.get("answer", "")),
            "num_sources": len(result.get("sources", [])),
            "source_titles": [s.get("title", "")[:60] for s in result.get("sources", [])],
            "agent_trace": result.get("agent_trace", []),
            "elapsed_seconds": elapsed_seconds,
        }

        # Append to JSONL file (one JSON object per line)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent_logs(self, n: int = 10) -> list[dict]:
        """Get the N most recent log entries."""
        if not os.path.exists(self.log_file):
            return []

        entries = []
        with open(self.log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries[-n:]

    def get_stats(self) -> dict:
        """Get aggregate statistics from all logged runs."""
        entries = self.get_recent_logs(n=10000)

        if not entries:
            return {"total_runs": 0}

        routes = [e["route"] for e in entries]
        scores = [e["quality_score"] for e in entries if e["quality_score"] > 0]
        times = [e["elapsed_seconds"] for e in entries if e.get("elapsed_seconds")]

        from collections import Counter
        return {
            "total_runs": len(entries),
            "route_distribution": dict(Counter(routes)),
            "avg_quality_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "avg_response_time": round(sum(times) / len(times), 2) if times else 0,
            "hyde_usage_rate": round(
                sum(1 for e in entries if e.get("hyde_used")) / len(entries), 2
            ),
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print("  AGENT TRACE STATISTICS")
        print(f"{'='*50}")
        print(f"  Total runs:         {stats['total_runs']}")
        print(f"  Avg quality score:  {stats.get('avg_quality_score', 'N/A')}/5")
        print(f"  Avg response time:  {stats.get('avg_response_time', 'N/A')}s")
        print(f"  HyDE usage rate:    {stats.get('hyde_usage_rate', 'N/A')}")
        print(f"  Route distribution: {stats.get('route_distribution', {})}")


if __name__ == "__main__":
    tracer = AgentTracer()
    tracer.print_stats()