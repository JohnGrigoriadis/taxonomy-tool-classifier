"""
Classify all tools with all four models sequentially.

Run:  uv run python scripts/run_classification.py [--smoke-test]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_taxonomy import config
from ai_taxonomy.pipeline.classify import run_classification
from ai_taxonomy.models.claude import ClaudeClassifier
from ai_taxonomy.models.openai_model import OpenAIClassifier
from ai_taxonomy.models.gemini_model import GeminiClassifier
from ai_taxonomy.models.mistral_model import MistralClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Classify only first 20 tools.")
    args = parser.parse_args()

    classifiers = [
        ClaudeClassifier(),
        OpenAIClassifier(),
        GeminiClassifier(),
        MistralClassifier(),
    ]

    tools_path = config.TOOLS_PATH
    if not tools_path.exists():
        print(f"Tools file not found at {tools_path}. Run scripts/download_data.py first.")
        sys.exit(1)

    if args.smoke_test:
        import tempfile, json as _json
        from ai_taxonomy.data.loader import load_tools
        tools = load_tools(tools_path)[:20]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for t in tools:
            tmp.write(_json.dumps(t) + "\n")
        tmp.flush()
        tools_path = Path(tmp.name)
        print(f"Smoke test: classifying {len(tools)} tools.")

    for clf in classifiers:
        print(f"\n{'='*50}\nRunning {clf}")
        run_classification(clf, tools_path=tools_path)

    print("\nAll classifiers done.")


if __name__ == "__main__":
    main()
