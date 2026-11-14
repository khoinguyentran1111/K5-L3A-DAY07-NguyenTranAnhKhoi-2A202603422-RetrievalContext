"""Retrieval/context benchmark on one split for several pipelines (no LLM answer generation).

Usage:
  python -m src.day7.eval_retrieval --split dev --pipelines dense bm25 hybrid_rrf --output results/retrieval/dev

For each answerable case: retrieve top-10 with the ORIGINAL question (+ the user-provided filter), mark hits with
the shared rule in data.chunk_hit, pack the top-3 into a context budget, and record latency and cost.
Gold is read only after retrieval to score it; it never reaches a retriever or a prompt.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from . import config
from .context import Neighbours, pack
from .data import chunk_hit, load_cases, load_chunks, write_json
from .pipelines import PIPELINES, Toolkit
from .retrievers import brief

RANK_DEPTH = 10
CONTEXT_TOP = 3


def evaluate(toolkit: Toolkit, name: str, cases: list[dict], budget: int,
             expand_top: int = 0) -> tuple[list[dict], dict]:
    retriever = toolkit.build(name)
    toolkit.embedder.clear_cache()   # each pipeline pays its own embedding latency/cost
    neighbours = Neighbours(toolkit.chunks) if expand_top else None
    traces = []
    for case in cases:
        if not case["answerable"]:
            continue   # unanswerable cases are excluded from retrieval metric denominators
        cost0 = toolkit.cost_snapshot()["cost_usd"]
        started = time.perf_counter()
        error = None
        try:
            results = retriever.search(case["question"], filters=case.get("filter"), top_k=RANK_DEPTH)
        except Exception as exc:   # report, never replace with fake results
            results, error = [], f"{exc.__class__.__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000
        context, ids = pack(results[:CONTEXT_TOP], budget, neighbours=neighbours, expand_top=expand_top)
        context_chunks = [toolkit.by_id[i] for i in ids]
        hits = [chunk_hit(r, case["gold"]) for r in results]
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        traces.append({
            "query_id": case["id"], "split": case["split"], "type": case["type"], "question": case["question"],
            "pipeline": name, "filter": case.get("filter"), "steps": retriever.last_steps if not error else [],
            "results": [{**b, "hit": h} for b, h in zip(brief(results), hits)],
            "context_ids": ids, "context_chars": len(context),
            "hit1": bool(hits[:1] and hits[0]), "hit3": any(hits[:3]), "hit5": any(hits[:5]),
            "first_hit_rank": first, "rr": 1 / first if first else 0.0,
            "context_hit": any(chunk_hit(c, case["gold"]) for c in context_chunks),
            "latency_ms": round(elapsed, 1), "cost_usd": round(toolkit.cost_snapshot()["cost_usd"] - cost0, 7),
            "error": error, "gold": case["gold"],
        })
    n = len(traces)
    summary = {
        "pipeline": name, "description": PIPELINES[name], "split": cases[0]["split"] if cases else None, "N": n,
        "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
        "hit5": sum(t["hit5"] for t in traces), "mrr10": round(sum(t["rr"] for t in traces) / n, 4),
        "context_hit": sum(t["context_hit"] for t in traces),
        "errors": sum(t["error"] is not None for t in traces),
        "context_top": CONTEXT_TOP, "expand_neighbours_of_top": expand_top, "budget_chars": budget,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
    }
    return traces, summary


def fmt(s: dict) -> str:
    N = s["N"]
    pct = lambda k: f"{s[k]}/{N} ({100 * s[k] / N:.0f}%)"
    label = s['pipeline'] + (f" +neighbours(top{s['expand_neighbours_of_top']})" if s['expand_neighbours_of_top'] else "")
    return (f"| {label} | {pct('hit1')} | {pct('hit3')} | {pct('hit5')} | {s['mrr10']:.3f} | "
            f"{pct('context_hit')} | {s['avg_context_chars']:.0f} | {s['median_latency_ms']:.0f} | "
            f"{s['total_cost_usd']:.5f} | {s['errors']} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--pipelines", nargs="+", required=True, choices=sorted(PIPELINES))
    parser.add_argument("--budget", type=int, default=3000, help="context budget in characters")
    parser.add_argument("--expand-top", type=int, default=0,
                        help="add prev/next chunks of the top-N results to the context (0 = off)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.split)
    toolkit = Toolkit(load_chunks())
    summaries = []
    for name in args.pipelines:
        traces, summary = evaluate(toolkit, name, cases, args.budget, args.expand_top)
        write_json(args.output / f"{name}.traces.json", traces)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    header = ("| Pipeline | Hit@1 | Evidence Hit@3 | Hit@5 | MRR@10 | Context Evidence Rate (top-3) | "
              "Avg context chars | Median latency ms | Cost USD (all queries) | Errors |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    table = header + "\n" + "\n".join(fmt(s) for s in summaries)
    write_json(args.output / "summary.json", {"split": args.split, "agent_model_for_llm_steps": config.AGENT_MODEL,
                                              "embedding_model": config.EMBEDDING_MODEL, "pipelines": summaries})
    (args.output / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
