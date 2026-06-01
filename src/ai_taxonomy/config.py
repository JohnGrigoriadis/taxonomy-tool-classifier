from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

# Paths
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
TAXONOMY_PATH = ROOT / "data" / "taxonomy.json"
GOLDEN_SET_PATH = DATA_PROCESSED / "golden_set.jsonl"
TOOLS_PATH = DATA_PROCESSED / "tools.jsonl"

OUTPUTS = ROOT / "outputs"
CLASSIFICATIONS_DIR = OUTPUTS / "classifications"
EVALUATION_DIR = OUTPUTS / "evaluation"
FIGURES_DIR = OUTPUTS / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, CLASSIFICATIONS_DIR, EVALUATION_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Model identifiers
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.0-flash"
MISTRAL_MODEL = "mistral-small-latest"

# Pipeline
BATCH_SIZE = 20
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # seconds between batches

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
