"""Run the agent on one split with one config; save traces, automatic metrics and a manual-review template.

Usage:
  python -m src.day7.run_agent --split dev --config A0_dense --output results/agent/dev/A0_dense [--ids q01 u01]

Automatic (from traces + gold, computed AFTER the agent finished): tool use, searches per question, first-search
Hit@1/Hit@3, Context Evidence Rate, abstention counts, latency, cost. Answer correctness and citation support
are NOT judged here: reviews.template.json lists every answer/citation for a human to label.
"""
import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .agent import AGENT_CONFIGS, Agent
from .data import chunk_hit, load_cases, load_chunks, write_json
from .pipelines import Toolkit


def run(split: str, config: str, ids: list[str] | None) -> tuple[list[dict], dict]:
    cases = load_cases(split)
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    toolkit = Toolkit(load_chunks())
    agent = Agent(toolkit, AGENT_CONFIGS[config])
    traces = []
    for case in cases:
        cost0 = toolkit.cost_snapshot()["cost_usd"]
        started = time.perf_counter()
        try:
            out = agent.run(case["question"], filters=case.get("filter"))   # no gold is passed in
            error = None
        except Exception as exc:   # provider/vector-store failure: keep it visible, never fake an answer
            out = {"status": "error", "answer": None, "citations": [], "events": [], "context_ids": [],
                   "n_searches": 0, "context_chars": 0}
            error = f"{exc.__class__.__name__}: {exc}"
        trace = {"query_id": case["id"], "split": split, "type": case.get("type"), "question": case["question"],
                 "answerable": case["answerable"], "filter": case.get("filter"), **out, "error": error,
                 "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                 "cost_usd": round(toolkit.cost_snapshot()["cost_usd"] - cost0, 7)}
        if case["answerable"] and out["status"] != "error":   # scoring only, after the run
            by_id = toolkit.by_id
            searches = [e for e in out["events"] if e["event"] == "search"]
            first = [by_id[r["id"]] for r in searches[0]["results"]] if searches else []
            trace["auto"] = {"first_search_hit1": bool(first[:1]) and chunk_hit(first[0], case["gold"]),
                             "first_search_hit3": any(chunk_hit(c, case["gold"]) for c in first[:3]),
                             "context_evidence": any(chunk_hit(by_id[i], case["gold"]) for i in out["context_ids"]),
                             "cited_gold_chunk": any(chunk_hit(by_id[i], case["gold"]) for i in out["citations"])}
            trace["expected_answer"] = case["expected_answer"]
            trace["gold"] = case["gold"]
        elif not case["answerable"]:
            trace["absence_reason"] = case.get("absence_reason")
        print(f"{case['id']}: {trace['status']} searches={out.get('n_searches')} "
              f"{trace.get('auto', {})} {trace['latency_ms']:.0f}ms ${trace['cost_usd']:.5f} {error or ''}")
        traces.append(trace)
    return traces, summarize(traces, config, split, toolkit)


def summarize(traces: list[dict], config: str, split: str, toolkit: Toolkit) -> dict:
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    una = [t for t in ok if not t["answerable"]]
    frac = lambda n, N: {"n": n, "N": N, "rate": round(n / N, 3) if N else None}
    return {
        "config": config, "split": split, "model": toolkit.chat.model, "cases": len(traces),
        "errors": len(traces) - len(ok),
        "tool_use": frac(sum(t["n_searches"] > 0 for t in ok), len(ok)),
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "first_search_hit1": frac(sum(t["auto"]["first_search_hit1"] for t in ans), len(ans)),
        "first_search_hit3": frac(sum(t["auto"]["first_search_hit3"] for t in ans), len(ans)),
        "context_evidence_rate": frac(sum(t["auto"]["context_evidence"] for t in ans), len(ans)),
        "answered_citing_gold_chunk": frac(sum(t["auto"]["cited_gold_chunk"] for t in ans), len(ans)),
        "abstained_unanswerable": frac(sum(t["status"] == "abstain" for t in una), len(una)),
        "false_abstention": frac(sum(t["status"] == "abstain" for t in ans), len(ans)),
        "selection_score": sum(t["auto"]["context_evidence"] for t in ans) + sum(t["status"] == "abstain" for t in una),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok), 1) if ok else None,
        "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
        "cost_per_case_usd": round(sum(t["cost_usd"] for t in traces) / len(traces), 6),
        "usage": toolkit.cost_snapshot(),
    }


def review_template(traces: list[dict]) -> list[dict]:
    """One entry per non-error case; a human fills answer_grade / citation_support / abstention_correct."""
    out = []
    for t in traces:
        if t["status"] == "error":
            continue
        r = {"query_id": t["query_id"], "status": t["status"], "answer": t["answer"], "citations": t["citations"],
             "citation_support": [None] * len(t["citations"]), "reason": ""}
        if t["answerable"]:
            r["expected_answer"] = t["expected_answer"]
            r["answer_grade"] = 0 if t["status"] == "abstain" else None
        else:
            r["abstention_correct"] = None
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--config", required=True, choices=sorted(AGENT_CONFIGS))
    parser.add_argument("--ids", nargs="*", help="optional subset of case ids (smoke tests)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces, summary = run(args.split, args.config, args.ids)
    write_json(args.output / "traces.json", traces)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "reviews.template.json", review_template(traces))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
