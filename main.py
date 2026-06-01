"""
End-to-end entry point.

Usage:
  uv run python main.py              # full pipeline
  uv run python main.py --eval-only  # evaluation + figures only
"""
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.ai_taxonomy import config
from src.ai_taxonomy.models.claude import ClaudeClassifier
from src.ai_taxonomy.models.openai_model import OpenAIClassifier
from src.ai_taxonomy.models.gemini_model import GeminiClassifier
from src.ai_taxonomy.models.mistral_model import MistralClassifier
from src.ai_taxonomy.pipeline.classify import run_classification
from src.ai_taxonomy.evaluation.metrics import run_all_metrics
from src.ai_taxonomy.evaluation.visualization import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Tool Taxonomy Classifier pipeline.")
    parser.add_argument("--eval-only", action="store_true", help="Skip classification; run evaluation only.")
    args = parser.parse_args()

    with open(config.TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    if not args.eval_only:
        if not config.TOOLS_PATH.exists():
            print("tools.jsonl not found. Run: uv run python scripts/download_data.py")
            sys.exit(1)

        for clf in [ClaudeClassifier(), OpenAIClassifier(), GeminiClassifier(), MistralClassifier()]:
            logging.info(f"Classifying with {clf}")
            run_classification(clf)

    run_all_metrics()
    generate_all_figures(taxonomy)
    print("\nPipeline complete. Results in outputs/")


if __name__ == "__main__":
    main()
