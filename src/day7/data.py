"""Load the Day 7 JSON contract and define one shared evidence-hit rule for every method."""
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from . import config

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def norm(text: str) -> str:
    """NFC, unify quotes/dashes, collapse whitespace. Case is kept so matches stay verbatim."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).translate(_QUOTES)).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_chunks() -> list[dict[str, Any]]:
    return load_json(config.CHUNKS_PATH)


def load_cases(split: str) -> list[dict[str, Any]]:
    """All benchmark cases of one split (answerable + unanswerable). Gold stays in the evaluator only."""
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    return [c for c in load_json(config.CASES_PATH) if c["split"] == split]


def chunk_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """A chunk is a hit if it comes from the gold source, its page range covers the gold page and it
    contains one gold evidence string completely (after whitespace/quote normalisation).

    Gold items are alternatives: any one of them is enough (all Day 6 questions are single-evidence).
    """
    page_end = chunk.get("metadata", {}).get("page_end", chunk["page"])
    text = norm(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] <= g["page"] <= page_end
               and norm(g["evidence"]) in text for g in gold)


def any_hit(chunks: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    return any(chunk_hit(c, gold) for c in chunks)
