"""Multi-agent workflow built with LangGraph.

planner -> researcher -> writer -> reviewer -> (revise -> researcher | end)
"""
import re
import time
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from .llm import get_llm
from .retriever import Retriever
from .tools import ALLOWED_TOOLS

MAX_REVISIONS = 1
BLOCKED = re.compile(r"ignore (all|previous) instructions|system prompt|password", re.I)


class State(TypedDict, total=False):
    question: str
    plan: List[str]
    evidence: List[Dict]
    answer: str
    review: str
    revisions: int
    trace: List[str]
    k: int


def build_graph(llm=None, retriever=None):
    llm = llm or get_llm()
    retriever = retriever or Retriever()

    def log(s, msg):
        return s.get("trace", []) + [msg]

    def planner(s: State):
        if BLOCKED.search(s["question"]):
            return {"plan": [], "trace": log(s, "planner: request blocked by safety filter")}
        steps = [x for x in llm.invoke("[PLAN]\n" + s["question"]).splitlines() if x.strip()]
        return {"plan": steps, "revisions": s.get("revisions", 0),
                "trace": log(s, f"planner: {len(steps)} sub-tasks")}

    def researcher(s: State):
        k = s.get("k", 3) + 2 * s.get("revisions", 0)  # widen search on revision
        ev, seen = [], set()
        for step in s["plan"]:
            if re.fullmatch(r"[\d\s\.\+\-\*/\(\)]+", step):  # pure arithmetic -> tool
                ev.append({"text": f"{step} = {ALLOWED_TOOLS['calculator'](step)}", "source": "calculator"})
                continue
            for hit in retriever.search(step, k):
                if hit["text"] not in seen:
                    seen.add(hit["text"]); ev.append(hit)
        return {"evidence": ev, "trace": log(s, f"researcher: {len(ev)} evidence chunks (k={k})")}

    def writer(s: State):
        ctx = "\n".join(f"[{e['source']}] {e['text'].strip()}" for e in s["evidence"])
        ans = llm.invoke(f"[WRITE]\nQuestion: {s['question']}\nUse only the context and cite sources.\nCONTEXT:\n{ctx}")
        return {"answer": ans, "trace": log(s, "writer: draft produced")}

    def reviewer(s: State):
        ok = bool(s["evidence"]) and re.search(r"\[[\w\.\-]+\]|\(.*?\.txt\)", s["answer"]) is not None
        return {"review": "pass" if ok else "revise", "trace": log(s, f"reviewer: {'pass' if ok else 'revise'}")}

    def bump(s: State):
        return {"revisions": s.get("revisions", 0) + 1}

    def after_plan(s): return "researcher" if s["plan"] else END
    def after_review(s): return END if s["review"] == "pass" or s.get("revisions", 0) >= MAX_REVISIONS else "bump"

    g = StateGraph(State)
    for n, f in [("planner", planner), ("researcher", researcher), ("writer", writer),
                 ("reviewer", reviewer), ("bump", bump)]:
        g.add_node(n, f)
    g.set_entry_point("planner")
    g.add_conditional_edges("planner", after_plan)
    g.add_edge("researcher", "writer"); g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", after_review)
    g.add_edge("bump", "researcher")
    return g.compile()


def ask(question: str, graph=None, k: int = 3):
    graph = graph or build_graph()
    t = time.perf_counter()
    out = graph.invoke({"question": question, "k": k, "trace": []})
    out["latency_s"] = time.perf_counter() - t
    return out
