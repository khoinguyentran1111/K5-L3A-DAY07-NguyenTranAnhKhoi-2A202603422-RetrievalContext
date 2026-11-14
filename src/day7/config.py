"""Paths, environment and fixed experiment settings. Everything a run depends on is listed here."""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# Day 6 project (sibling folder by default); only read, never modified.
DAY6_ROOT = Path(os.getenv("DAY6_ROOT", REPO_ROOT.parent / "K5-L3A-DAY06-2A202603422-DataIngestion"))
DAY6_CHUNKS = DAY6_ROOT / "results" / "chunks" / "B.jsonl"
DAY6_PAGES = DAY6_ROOT / "results" / "pages.jsonl"
DAY6_QUESTIONS = DAY6_ROOT / "benchmark" / "questions.json"

DATA_DIR = REPO_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "corpus" / "chunks.json"
QUESTIONS_PATH = DATA_DIR / "benchmark" / "questions.json"      # answerable only (Day 7 schema)
CASES_PATH = DATA_DIR / "benchmark" / "cases.json"              # all cases incl. unanswerable
UNANSWERABLE_PATH = DATA_DIR / "benchmark" / "unanswerable.json"  # hand-written, verified absent
RESULTS_DIR = REPO_ROOT / "results"

SOURCE_ID = "openstax-college-success"
SOURCE = "OpenStax College Success (CC BY-NC-SA 4.0), https://openstax.org/details/books/college-success"
DAY6_STRATEGY = "B"   # Day 6 selected index: heading/paragraph chunks, max 800 chars

DATABASE_URL = os.getenv("DATABASE_URL", "").strip().strip('"')
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip().strip('"')
AGENT_MODEL = os.getenv("AGENT_MODEL", "anthropic/claude-haiku-5.5").strip().strip('"')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_DIMENSIONS = 1536

FILTERABLE_FIELDS = ("chapter", "section")

# Pre-registered selection rules (written before any Day 7 dev run, never changed after looking at test).
RETRIEVAL_SELECTION_RULE = (
    "1) highest dev Evidence Hit@3 on the original question pipeline; 2) tie -> highest dev Hit@1; "
    "3) tie -> lower cost per query, then lower median latency.")
AGENT_SELECTION_RULE = (
    "1) highest dev (Context Evidence Rate on answerable + correct abstentions on unanswerable), counted "
    "automatically from traces; 2) tie -> fewer false abstentions; 3) tie -> lower cost per query; "
    "the chosen config and the baseline are then graded manually on dev before freezing.")
