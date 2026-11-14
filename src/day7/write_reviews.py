"""Write reviews.json from compact manual labels (alias -> supported?) so labels stay readable and auditable.

Usage: python -m src.day7.write_reviews <run_dir> <labels.json>
labels.json: {"reviewer": "...", "method": "...", "labels": {"q01": {"grade": 2, "support": {"C1": true}, "reason": "..."},
              "u01": {"abstention_correct": true, "reason": "..."}}}
Citation support is given per alias the model cited; it is mapped onto trace["citations"] order.
"""
import sys
from pathlib import Path

from .data import load_json, write_json


def main() -> None:
    run_dir, labels_path = Path(sys.argv[1]), Path(sys.argv[2])
    traces = load_json(run_dir / "traces.json")
    spec = load_json(labels_path)
    labels = spec["labels"]
    reviews = []
    for t in traces:
        if t["status"] == "error":
            continue
        lab = labels[t["query_id"]]
        alias_of = {cid: a for a, cid in t.get("alias_map", {}).items()}
        support = [lab["support"][alias_of[cid]] for cid in t["citations"]]
        r = {"query_id": t["query_id"], "status": t["status"], "citations": t["citations"],
             "citation_support": support, "reason": lab["reason"], "reviewer": spec["reviewer"]}
        if t["answerable"]:
            r["answer_grade"] = lab["grade"]
        else:
            r["abstention_correct"] = lab["abstention_correct"]
        reviews.append(r)
    extra = sorted(set(labels) - {t["query_id"] for t in traces})
    if extra:
        raise ValueError(f"labels for unknown ids: {extra}")
    write_json(run_dir / "reviews.json", reviews)
    print(f"wrote {len(reviews)} reviews to {run_dir / 'reviews.json'}")


if __name__ == "__main__":
    main()
