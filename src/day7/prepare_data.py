"""Adapter: Day 6 strategy-B chunks + benchmark -> Day 7 JSON contract.

Usage:  python -m src.day7.prepare_data [--check-db]

* chunks.json: id, text, source_id, source, page, metadata (chapter, section, page_end, ...).
* questions.json: answerable questions with gold = main evidence + Day 6 alternatives.
* cases.json: answerable + hand-written unanswerable cases, the single file every runner reads.
Gold items that do not fit inside one chunk of the right page are dropped and reported, never edited.
"""
import argparse
import hashlib
import json
from collections import Counter

from . import config
from .data import chunk_hit, load_json, write_json


def convert_chunks() -> list[dict]:
    rows = [json.loads(line) for line in config.DAY6_CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = []
    for r in rows:
        chunks.append({
            "id": r["chunk_id"], "text": r["text"], "source_id": config.SOURCE_ID, "source": config.SOURCE,
            "page": r["page"],
            "metadata": {"page_end": r["page_end"], "printed_page": r.get("printed_page"), "chapter": r["chapter"],
                         "chapter_title": r["chapter_title"], "section": r["section"],
                         "char_start": r["char_start"], "char_end": r["char_end"],
                         "day6_strategy": config.DAY6_STRATEGY, "embedding_model": config.EMBEDDING_MODEL},
        })
    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids in Day 6 export")
    return chunks


def convert_questions(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    raw = load_json(config.DAY6_QUESTIONS)["questions"]
    questions, report = [], []
    for q in raw:
        candidates = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
        candidates += [{"source_id": a["gold_source_id"], "page": a["gold_page"], "evidence": a["evidence_quote"]}
                       for a in q.get("alternatives") or []]
        gold, dropped = [], []
        for g in candidates:
            (gold if any(chunk_hit(c, [g]) for c in chunks) else dropped).append(g)
        if not gold:
            raise ValueError(f"{q['id']}: no gold evidence fits inside one strategy-B chunk")
        questions.append({"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                          "expected_answer": q["expected_answer"], "gold": gold, "filter": q.get("filter")})
        report.append({"id": q["id"], "gold_kept": len(gold), "gold_dropped": dropped,
                       "gold_chunks": [c["id"] for c in chunks if chunk_hit(c, gold)]})
    return questions, report


def check_db(chunks: list[dict]) -> dict:
    """Confirm the JSON chunks are exactly the rows (id + text hash + model) indexed in pgvector."""
    import psycopg
    with psycopg.connect(config.DATABASE_URL, connect_timeout=5) as conn:
        rows = conn.execute("SELECT chunk_id, content_hash, embedding_model, vector_dims(embedding) FROM chunks "
                            "WHERE strategy = %s", (config.DAY6_STRATEGY,)).fetchall()
        versions = conn.execute("SELECT current_setting('server_version'), extversion FROM pg_extension "
                                "WHERE extname='vector'").fetchone()
    db = {r[0]: r for r in rows}
    mismatched = [c["id"] for c in chunks if c["id"] not in db
                  or db[c["id"]][1] != hashlib.sha256(c["text"].encode("utf-8")).hexdigest()]
    return {"db_rows": len(rows), "json_chunks": len(chunks), "missing_or_changed": mismatched,
            "extra_in_db": sorted(set(db) - {c["id"] for c in chunks}),
            "embedding_models": sorted({r[2] for r in rows}), "dimensions": sorted({r[3] for r in rows}),
            "postgres": versions[0], "pgvector": versions[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-db", action="store_true", help="also compare chunk ids/hashes with pgvector")
    args = parser.parse_args()
    chunks = convert_chunks()
    questions, report = convert_questions(chunks)
    unanswerable = load_json(config.UNANSWERABLE_PATH)
    cases = [{**q, "answerable": True, "expected_behavior": "search_and_answer"} for q in questions] + unanswerable
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case ids")
    write_json(config.CHUNKS_PATH, chunks)
    write_json(config.QUESTIONS_PATH, questions)
    write_json(config.CASES_PATH, cases)
    summary = {
        "chunks": len(chunks), "pages": len({c["page"] for c in chunks}),
        "split_counts": Counter(f"{c['split']}/{'answerable' if c['answerable'] else 'unanswerable'}" for c in cases),
        "gold_dropped": {r["id"]: r["gold_dropped"] for r in report if r["gold_dropped"]},
        "gold_chunk_count": {r["id"]: len(r["gold_chunks"]) for r in report},
    }
    if args.check_db:
        summary["db_check"] = check_db(chunks)
    write_json(config.DATA_DIR / "benchmark" / "prepare_report.json", {**summary, "per_question": report})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
