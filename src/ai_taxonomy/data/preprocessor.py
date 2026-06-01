from .loader import clean_description

MIN_DESC_LENGTH = 30


def normalize_tool(tool: dict) -> dict:
    return {
        "id": tool["id"],
        "name": tool.get("name", "").strip(),
        "description": clean_description(tool.get("description", "")),
        "source": tool.get("source", "unknown"),
    }


def is_valid(tool: dict) -> bool:
    return len(tool.get("description", "")) >= MIN_DESC_LENGTH and bool(tool.get("name"))


def deduplicate(tools: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for tool in tools:
        key = tool["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(tool)
    return unique


def preprocess(tools: list[dict]) -> list[dict]:
    normalized = [normalize_tool(t) for t in tools]
    valid = [t for t in normalized if is_valid(t)]
    return deduplicate(valid)
