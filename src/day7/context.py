"""Context selection: dedup by id, keep rank order, optional neighbour expansion, hard character budget.

Budget unit = characters of the packed context string (headers included). A chunk that does not fit is
skipped (not truncated) so every cited chunk is shown in full.
"""
from typing import Any


def block(c: dict[str, Any]) -> str:
    m = c["metadata"]
    pages = f"{c['page']}" if m.get("page_end", c["page"]) == c["page"] else f"{c['page']}-{m['page_end']}"
    return f"[{c['id']}] (page {pages}; {m.get('section')})\n{c['text']}"


class Neighbours:
    """Previous/next chunk in reading order (same chapter), for 'keep adjacent passages' experiments."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        order = sorted(chunks, key=lambda c: (c["metadata"]["chapter"], c["metadata"]["char_start"]))
        self.prev: dict[str, dict[str, Any]] = {}
        self.next: dict[str, dict[str, Any]] = {}
        for a, b in zip(order, order[1:]):
            if a["metadata"]["chapter"] == b["metadata"]["chapter"]:
                self.next[a["id"]], self.prev[b["id"]] = b, a

    def around(self, chunk_id: str) -> list[dict[str, Any]]:
        return [c for c in (self.prev.get(chunk_id), self.next.get(chunk_id)) if c]


def pack(chunks: list[dict[str, Any]], budget: int, *, neighbours: Neighbours | None = None,
         expand_top: int = 0) -> tuple[str, list[str]]:
    """Return (context, ids). With neighbours, the top ``expand_top`` hits get prev/next chunks right after them."""
    ordered: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        ordered.append(c)
        if neighbours and i < expand_top:
            ordered.extend(neighbours.around(c["id"]))
    blocks, ids, seen, used = [], [], set(), 0
    for c in ordered:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        b = block(c)
        extra = len(b) + (2 if blocks else 0)
        if used + extra <= budget:
            blocks.append(b)
            ids.append(c["id"])
            used += extra
    return "\n\n".join(blocks), ids
