"""Offline tests for src/day7 (code behaviour only; fakes here are never used for benchmark numbers)."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.day7.agent import Agent, AgentConfig
    from src.day7.context import Neighbours, pack
    from src.day7.data import chunk_hit, norm
    from src.day7.retrievers import BM25Retriever, HybridRRF, tokenize
except ImportError as exc:  # openai/psycopg missing: run with the project venv
    raise unittest.SkipTest(f"day7 deps not installed ({exc}); use .venv/Scripts/python") from exc


def chunk(cid, text, page=1, chapter=3, start=0, section="3.1 A", page_end=None):
    return {"id": cid, "text": text, "source_id": "doc", "source": "doc://", "page": page,
            "metadata": {"chapter": chapter, "section": section, "char_start": start,
                         "page_end": page_end if page_end is not None else page}}


CHUNKS = [
    chunk("c1", "The Pomodoro Technique uses a timer for 25 minute intervals.", page=105, start=0),
    chunk("c2", "After four pomodoros take a longer break of 20 to 30 minutes.", page=105, start=100),
    chunk("c3", "Memory has three steps: encoding, storage and retrieval.", page=183, chapter=6, start=0,
          section="6.1 Memory"),
]


class HitRuleTests(unittest.TestCase):
    def test_hit_requires_source_page_and_full_evidence(self):
        gold = [{"source_id": "doc", "page": 105, "evidence": "longer  break of 20\nto 30 minutes"}]
        self.assertTrue(chunk_hit(CHUNKS[1], gold))          # whitespace-normalised match
        self.assertFalse(chunk_hit(CHUNKS[0], gold))         # wrong text
        self.assertFalse(chunk_hit(CHUNKS[1], [{**gold[0], "page": 106}]))   # wrong page
        self.assertFalse(chunk_hit(CHUNKS[1], [{**gold[0], "source_id": "x"}]))

    def test_cross_page_chunk_covers_page_range(self):
        c = chunk("x", "Average total loss: $151,115", page=77, page_end=78)
        self.assertTrue(chunk_hit(c, [{"source_id": "doc", "page": 78, "evidence": "Average total loss: $151,115"}]))

    def test_norm_unifies_quotes(self):
        self.assertEqual(norm("“eat the frog”"), '"eat the frog"')


class RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_and_filters(self):
        bm25 = BM25Retriever(CHUNKS)
        res = bm25.search("longer break after four pomodoros", top_k=2)
        self.assertEqual(res[0]["id"], "c2")
        self.assertEqual([r["rank"] for r in res], list(range(1, len(res) + 1)))
        self.assertEqual(bm25.search("pomodoro timer", filters={"chapter": 6}, top_k=3), [])
        with self.assertRaises(ValueError):
            bm25.search("x", filters={"page": 1}, top_k=3)
        with self.assertRaises(ValueError):
            bm25.search("  ", top_k=3)

    def test_tokenize_stems_and_drops_stopwords(self):
        self.assertEqual(tokenize("The breaks are studying taking"), ["break", "study", "taking"])  # short words kept

    def test_rrf_fuses_by_rank(self):
        class Fixed:
            def __init__(self, order):
                self.order, self.last_steps = order, []

            def search(self, q, filters=None, top_k=3):
                by = {c["id"]: c for c in CHUNKS}
                return [{**by[i], "rank": r + 1, "score": 1.0} for r, i in enumerate(self.order[:top_k])]

        hybrid = HybridRRF(Fixed(["c1", "c2", "c3"]), Fixed(["c2", "c3", "c1"]), k=60, depth=3)
        self.assertEqual([r["id"] for r in hybrid.search("q", top_k=3)], ["c2", "c1", "c3"])


class ContextTests(unittest.TestCase):
    def test_pack_dedups_and_respects_budget(self):
        ctx, ids = pack([CHUNKS[0], CHUNKS[0], CHUNKS[1]], budget=10_000)
        self.assertEqual(ids, ["c1", "c2"])
        ctx, ids = pack(CHUNKS, budget=len(ctx.split("\n\n")[0]) + 1)
        self.assertEqual(ids, ["c1"])

    def test_neighbours_stay_inside_chapter(self):
        n = Neighbours(CHUNKS)
        self.assertEqual([c["id"] for c in n.around("c1")], ["c2"])
        self.assertEqual(n.around("c3"), [])
        _, ids = pack([CHUNKS[0]], budget=10_000, neighbours=n, expand_top=1)
        self.assertEqual(ids, ["c1", "c2"])


# ------------------------------------------------------------------ agent loop with scripted model
def tool_call(query, cid="t1"):
    fn = SimpleNamespace(name="search_documents", arguments=json.dumps({"query": query}))
    return SimpleNamespace(id=cid, function=fn, model_dump=lambda: {"id": cid, "type": "function",
                                                                     "function": {"name": fn.name, "arguments": fn.arguments}})


class ScriptedChat:
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, **kw):
        self.calls.append({"messages": list(messages), **kw})
        content, calls = self.replies.pop(0)
        return SimpleNamespace(content=content, tool_calls=calls), {"cost_usd": 0.0}


class FakeRetriever:
    name, last_steps = "fake", []

    def search(self, query, filters=None, top_k=3):
        return [{**CHUNKS[1], "rank": 1, "score": 0.9}][:top_k]


def toolkit(chat):
    return SimpleNamespace(chunks=CHUNKS, chat=chat, build=lambda name: FakeRetriever())


class AgentTests(unittest.TestCase):
    def test_answer_with_valid_citation_maps_alias_to_chunk_id(self):
        chat = ScriptedChat([(None, [tool_call("pomodoro long break")]),
                             ('{"status":"answered","answer":"Nghỉ 20-30 phút [C1].","citations":["C1"]}', None)])
        out = Agent(toolkit(chat), AgentConfig("t", "fake")).run("Nghỉ bao lâu?")
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["c2"])
        self.assertIn("[c2]", out["answer"])
        self.assertEqual(out["n_searches"], 1)
        tool_msg = chat.calls[1]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("[C1]", tool_msg["content"])

    def test_citation_not_in_context_becomes_abstention(self):
        bad = '{"status":"answered","answer":"Sai [C9].","citations":["C9"]}'
        chat = ScriptedChat([(None, [tool_call("q")]), (bad, None), (bad, None)])   # repair turn fails too
        out = Agent(toolkit(chat), AgentConfig("t", "fake")).run("?")
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["events"][-1]["event"], "invalid_or_missing_citations")
        self.assertIn("repair_request", [e["event"] for e in out["events"]])
        self.assertIn("Sai [C9]", out["raw_final"])

    def test_repair_turn_fixes_missing_inline_citation(self):
        chat = ScriptedChat([(None, [tool_call("q")]),
                             ('{"status":"answered","answer":"Nghỉ 20-30 phút.","citations":["C1"]}', None),
                             ('{"status":"answered","answer":"Nghỉ 20-30 phút [C1].","citations":["C1"]}', None)])
        out = Agent(toolkit(chat), AgentConfig("t", "fake")).run("?")
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["c2"])
        self.assertEqual(chat.calls[-1]["tool_choice"], "none")
        self.assertIn("no inline", chat.calls[-1]["messages"][-1]["content"])

    def test_search_limit_forces_answer_and_dedups_context(self):
        chat = ScriptedChat([(None, [tool_call("a", "t1")]), (None, [tool_call("b", "t2")]),
                             ('{"status":"insufficient_evidence","answer":"Không có.","citations":[]}', None)])
        out = Agent(toolkit(chat), AgentConfig("t", "fake", max_searches=2)).run("?")
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["context_ids"], ["c2"])                 # second search repeated c2
        self.assertEqual(chat.calls[-1]["tool_choice"], "none")       # forced final answer
        self.assertIn("Already shown", chat.calls[-1]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
