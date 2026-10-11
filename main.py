import sys
from app.graph import ask

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is RAG and how do multi-agent systems reduce hallucination?"
    r = ask(q)
    print("\n".join(r["trace"]))
    print("\nANSWER:\n" + r.get("answer", "Request refused."))
    print(f"\nlatency: {r['latency_s']:.3f}s")
