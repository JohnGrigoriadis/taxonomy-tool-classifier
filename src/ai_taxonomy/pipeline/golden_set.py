import json
import random
from pathlib import Path
from .. import config
from ..data.loader import load_tools
from ..taxonomy.schema import ClassificationResult


def sample_for_review(
    classifications_path: Path,
    n: int = 50,
    seed: int = 42,
) -> list[dict]:
    results = []
    with open(classifications_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    random.seed(seed)
    return random.sample(results, min(n, len(results)))


def save_golden_entry(
    tool_id: str,
    correct_l3_id: str,
    model_was_wrong: bool,
    correction_note: str = "",
    path: Path = config.GOLDEN_SET_PATH,
) -> None:
    entry = {
        "tool_id": tool_id,
        "correct_l3_id": correct_l3_id,
        "model_was_wrong": model_was_wrong,
        "correction_note": correction_note,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_golden_set(path: Path = config.GOLDEN_SET_PATH) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_model_accuracy(results_path: Path, golden_path: Path = config.GOLDEN_SET_PATH) -> float:
    golden = {e["tool_id"]: e["correct_l3_id"] for e in load_golden_set(golden_path)}
    if not golden:
        return 0.0

    correct = total = 0
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["tool_id"] in golden:
                total += 1
                if r["l3_id"] == golden[r["tool_id"]]:
                    correct += 1
    return correct / total if total else 0.0
