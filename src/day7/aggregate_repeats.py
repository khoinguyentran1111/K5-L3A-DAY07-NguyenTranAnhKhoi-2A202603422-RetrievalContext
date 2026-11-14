"""Mean / min-max of retrieval metrics over repeated dev runs (LLM steps are not fully deterministic).

Usage: python -m src.day7.aggregate_repeats results/retrieval/dev results/retrieval/dev_repeat2 ... --output F.md
"""
import argparse
import json
from pathlib import Path

METRICS = ("hit1", "hit3", "hit5", "context_hit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_pipe: dict[str, list[dict]] = {}
    for run in args.runs:
        for s in json.loads((run / "summary.json").read_text(encoding="utf-8"))["pipelines"]:
            by_pipe.setdefault(s["pipeline"], []).append(s)
    lines = ["| Pipeline | Runs | N | " + " | ".join(f"{m} mean [min-max]" for m in METRICS) +
             " | MRR@10 mean | Cost/query USD | Median latency ms |",
             "|---" * (len(METRICS) + 6) + "|"]
    for name, rows in by_pipe.items():
        N = rows[0]["N"]
        cells = []
        for m in METRICS:
            vals = [r[m] for r in rows]
            cells.append(f"{sum(vals) / len(vals):.2f}/{N} [{min(vals)}-{max(vals)}]")
        mrr = sum(r["mrr10"] for r in rows) / len(rows)
        cost = sum(r["total_cost_usd"] for r in rows) / len(rows) / N
        lat = sum(r["median_latency_ms"] for r in rows) / len(rows)
        lines.append(f"| {name} | {len(rows)} | {N} | " + " | ".join(cells) + f" | {mrr:.3f} | {cost:.6f} | {lat:.0f} |")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
