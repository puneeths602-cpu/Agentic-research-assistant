"""Evaluates the pipeline: keyword coverage, groundedness, source hit rate, latency."""
import json
import re
import statistics
from app.graph import build_graph, ask

CASES = [
    {"q": "What is RAG and why does it reduce hallucination?", "kw": ["retriev", "chunk", "vector"], "src": "rag.txt"},
    {"q": "How do multi-agent systems work and what frameworks exist?", "kw": ["planner", "langgraph", "agent"], "src": "agentic_ai.txt"},
    {"q": "When should I fine-tune a model and what is LoRA?", "kw": ["lora", "fine-tun"], "src": "fine_tuning.txt"},
    {"q": "How can I reduce latency of an LLM application?", "kw": ["latency", "cach"], "src": "evaluation_safety.txt"},
    {"q": "What safety practices should agents follow?", "kw": ["tool", "approval", "log"], "src": "evaluation_safety.txt"},
]


def words(t): return set(re.findall(r"[a-z]{4,}", t.lower()))


def run():
    g = build_graph()
    rows = []
    for c in CASES:
        r = ask(c["q"], g)
        ans = r["answer"].lower()
        cov = sum(k in ans for k in c["kw"]) / len(c["kw"])
        ev_words = words(" ".join(e["text"] for e in r["evidence"]))
        grounded = len(words(r["answer"]) & ev_words) / max(1, len(words(r["answer"])))
        hit = any(e["source"] == c["src"] for e in r["evidence"])
        rows.append({"q": c["q"], "coverage": cov, "groundedness": round(grounded, 2),
                     "source_hit": hit, "latency_s": round(r["latency_s"], 3)})
    lat = [x["latency_s"] for x in rows]
    summary = {"n": len(rows),
               "avg_coverage": round(statistics.mean(x["coverage"] for x in rows), 2),
               "avg_groundedness": round(statistics.mean(x["groundedness"] for x in rows), 2),
               "source_hit_rate": round(sum(x["source_hit"] for x in rows) / len(rows), 2),
               "avg_latency_s": round(statistics.mean(lat), 3), "max_latency_s": max(lat)}
    print(json.dumps({"summary": summary, "cases": rows}, indent=2))


if __name__ == "__main__":
    run()
