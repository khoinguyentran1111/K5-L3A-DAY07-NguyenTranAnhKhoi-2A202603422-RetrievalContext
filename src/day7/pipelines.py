"""Named retrieval pipelines shared by the retrieval benchmark and the agent's search tool."""
from typing import Any

from .clients import Chat, Embedder
from .retrievers import BM25Retriever, DenseRetriever, HybridRRF, LLMRerank, QueryRewrite

PIPELINES = {
    "dense": "Dense cosine search on Day 6 pgvector (strategy B, text-embedding-3-small).",
    "bm25": "Okapi BM25 (k1=1.5, b=0.75) on the same chunks, English tokenizer + light stemming.",
    "hybrid_rrf": "RRF(k=60) of dense top-20 and BM25 top-20.",
    "rewrite_bm25": "LLM rewrites the question into an English query -> BM25 (isolates the language gap).",
    "rewrite_dense": "LLM rewrites the question into an English query -> dense.",
    "rewrite_hybrid": "LLM rewrites the question into an English query -> hybrid RRF.",
    "rewrite_hybrid_rerank": "rewrite_hybrid top-15 -> LLM listwise rerank (original question).",
    "hybrid_rerank": "hybrid RRF top-15 -> LLM listwise rerank (used by the agent, which writes English queries).",
}


class Toolkit:
    """Lazily builds shared clients so every pipeline uses the same embedder/LLM usage counters."""

    def __init__(self, chunks: list[dict[str, Any]], chat_model: str | None = None) -> None:
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.embedder = Embedder()
        self.chat = Chat(chat_model)
        self.dense = DenseRetriever(chunks, self.embedder)
        self.bm25 = BM25Retriever(chunks)

    def build(self, name: str):
        if name == "dense":
            return self.dense
        if name == "bm25":
            return self.bm25
        if name == "hybrid_rrf":
            return HybridRRF(self.dense, self.bm25)
        if name == "rewrite_bm25":
            return QueryRewrite(self.chat, self.bm25)
        if name == "rewrite_dense":
            return QueryRewrite(self.chat, self.dense)
        if name == "rewrite_hybrid":
            return QueryRewrite(self.chat, HybridRRF(self.dense, self.bm25))
        if name == "hybrid_rerank":
            return LLMRerank(self.chat, HybridRRF(self.dense, self.bm25))
        if name == "rewrite_hybrid_rerank":
            return LLMRerank(self.chat, QueryRewrite(self.chat, HybridRRF(self.dense, self.bm25)))
        raise ValueError(f"unknown pipeline {name!r}; choose from {sorted(PIPELINES)}")

    def cost_snapshot(self) -> dict[str, Any]:
        return {"embedding": self.embedder.usage.snapshot(), "chat": self.chat.usage.snapshot(),
                "cost_usd": round(self.embedder.usage.cost_usd + self.chat.usage.cost_usd, 6)}
