import json
import re
from pathlib import Path
from typing import Generator


def load_tools(path: Path) -> list[dict]:
    tools = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tools.append(json.loads(line))
    return tools


def clean_description(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"<[^>]+>", "", text)
    return text


def iter_tools(path: Path) -> Generator[dict, None, None]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
