# Multi-Agent Research Assistant (Agentic RAG)

A multi-agent workflow built with **LangGraph** that answers questions from a document knowledge base
using Retrieval-Augmented Generation (RAG) on a **ChromaDB** vector store, with a reviewer loop,
tool calling, safety guardrails and an evaluation harness.

## Architecture
```
question -> Planner -> Researcher -> Writer -> Reviewer --pass--> answer
                ^                                 |
                +------ widen search (revise) ----+   (max 1 revision)
```
- **Planner**: splits the question into sub-tasks (blocks unsafe/prompt-injection requests).
- **Researcher**: retrieves top-k chunks per sub-task from ChromaDB; calls allow-listed tools (safe calculator).
- **Writer**: drafts an answer using only retrieved context, with source citations.
- **Reviewer**: checks the answer is grounded and cited; triggers one revision with wider retrieval.
- **State/memory**: typed shared graph state + vector store as long-term memory.

## Run
```bash
pip install -r requirements.txt
python main.py "What is RAG and how do multi-agent systems work?"
python -m eval.evaluate      # coverage, groundedness, source hit rate, latency
pytest
```
Works offline with a deterministic extractive model. For real LLM output set `OPENAI_API_KEY`
(`pip install langchain-openai`) or `GOOGLE_API_KEY` (`pip install langchain-google-genai`).

## Safety
Input filter, allow-listed tools only, AST-based calculator (no `eval`), step limit on the revision loop,
full execution trace printed for auditing.

## Evaluation
`eval/evaluate.py` reports keyword coverage, groundedness (answer words supported by retrieved evidence),
source hit rate and latency. Re-run it with a real LLM and record your numbers here.

## Ideas to extend
Fine-tune a small model for the writer step, swap the hash embedder for sentence-transformers/OpenAI
embeddings, add a web-search tool, add a CrewAI/AutoGen comparison.
