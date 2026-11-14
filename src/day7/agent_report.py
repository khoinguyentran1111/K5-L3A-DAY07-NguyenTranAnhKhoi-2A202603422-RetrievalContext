"""Combine an agent run's traces with the human review labels via starter/agent_metrics.py.

Usage: python -m src.day7.agent_report results/agent/test/A2_hybrid_rerank_neighbours [more run dirs ...]

Each run dir needs traces.json (run_agent) and reviews.json (manual labels: answer_grade 0-2 for answerable,
abstention_correct for unanswerable, one citation_support boolean per citation, reason). Writes
agent_metrics.json next to them and prints one Markdown row per run.
"""
import argparse
import json
import sys
from pathlib import Path

from . import config
from .data import load_json, write_json

sys.path.insert(0, str(config.REPO_ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402  (starter script, unchanged)


def report(run_dir: Path) -> dict:
    traces = load_json(run_dir / "traces.json")
    reviews = load_json(run_dir / "reviews.json")
    summary = load_json(run_dir / "summary.json")
    missing = [r["query_id"] for r in reviews for k in ("answer_grade", "abstention_correct")
               if k in r and r[k] is None] + [r["query_id"] for r in reviews if None in r["citation_support"]]
    if missing:
        raise ValueError(f"{run_dir}: unlabeled reviews for {sorted(set(missing))}")
    cases = [{"id": t["query_id"], "split": t["split"], "answerable": t["answerable"]} for t in traces]
    slim = [{"query_id": t["query_id"], "status": t["status"], "events": t["events"], "citations": t["citations"]}
            for t in traces]
    metrics = aggregate(cases, slim, reviews)
    out = {"run": str(run_dir.relative_to(config.REPO_ROOT)), "config": summary["config"], "split": summary["split"],
           "model": summary["model"], "agent_metrics": metrics, "auto": {k: summary[k] for k in (
               "searches_distribution", "first_search_hit1", "first_search_hit3", "context_evidence_rate",
               "abstained_unanswerable", "false_abstention", "avg_context_chars", "median_latency_ms",
               "total_cost_usd", "cost_per_case_usd", "errors")}}
    write_json(run_dir / "agent_metrics.json", out)
    return out


def row(o: dict) -> str:
    m, a = o["agent_metrics"], o["auto"]
    g = m["grounded_answer"]
    f = lambda d: f"{d['n']}/{d['N']}"
    return (f"| {o['config']} / {o['split']} | {f(m['tool_use'])} | {a['searches_distribution']} | "
            f"{g['0']}/{g['1']}/{g['2']} (mean {g['mean']:.2f}/2) | {f(m['citation_validity'])} | "
            f"{f(m['answer_citation_coverage'])} | {f(m['abstention'])} | {f(m['false_abstention'])} | "
            f"{f(a['context_evidence_rate'])} | {a['median_latency_ms']/1000:.1f}s | ${a['total_cost_usd']:.4f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print("| Config / split | Tool use | Searches/case | Grounded 0/1/2 | Citation valid | Answers with citation | "
          "Abstention correct | False abstention | Context Evidence | Median latency | Total cost |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in args.runs:
        print(row(report(run.resolve())))


if __name__ == "__main__":
    main()
