import json
import time
import logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from .. import config
from ..data.loader import load_tools
from ..models.base import BaseClassifier
from ..taxonomy.schema import ClassificationResult

logger = logging.getLogger(__name__)


def _output_path(classifier: BaseClassifier) -> Path:
    safe_name = classifier.model_name.replace("/", "-").replace(":", "-")
    return config.CLASSIFICATIONS_DIR / f"{safe_name}_results.jsonl"


def _already_classified(output_path: Path) -> set[str]:
    done: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["tool_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _classify_one(classifier: BaseClassifier, tool: dict, taxonomy_json: str) -> ClassificationResult:
    return classifier.classify(
        tool_id=tool["id"],
        tool_name=tool["name"],
        tool_description=tool["description"],
        taxonomy_json=taxonomy_json,
    )


def run_classification(
    classifier: BaseClassifier,
    tools_path: Path = config.TOOLS_PATH,
    taxonomy_path: Path = config.TAXONOMY_PATH,
    batch_size: int = config.BATCH_SIZE,
) -> Path:
    tools = load_tools(tools_path)
    taxonomy_json = taxonomy_path.read_text()
    output_path = _output_path(classifier)
    done_ids = _already_classified(output_path)

    remaining = [t for t in tools if t["id"] not in done_ids]
    logger.info(f"{classifier}: {len(done_ids)} already done, {len(remaining)} remaining.")

    with open(output_path, "a") as out_f:
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            for tool in tqdm(batch, desc=f"{classifier.model_name} batch {i // batch_size + 1}"):
                try:
                    result = _classify_one(classifier, tool, taxonomy_json)
                    out_f.write(result.model_dump_json() + "\n")
                    out_f.flush()
                except Exception as e:
                    logger.error(f"Failed on {tool['id']}: {e}")
            time.sleep(config.RATE_LIMIT_DELAY)

    logger.info(f"Results written to {output_path}")
    return output_path
