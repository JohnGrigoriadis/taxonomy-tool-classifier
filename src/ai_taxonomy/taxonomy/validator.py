import json
from pathlib import Path
from typing import Any
from .schema import ClassificationResult


def load_taxonomy(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _collect_ids(nodes: list[dict], level: int) -> set[str]:
    ids: set[str] = set()
    for node in nodes:
        if node.get("level") == level:
            ids.add(node["id"])
        for child in node.get("children", []):
            if isinstance(child, dict):
                ids |= _collect_ids([child], level)
    return ids


def build_valid_ids(taxonomy: dict) -> dict[int, set[str]]:
    nodes = taxonomy["nodes"]
    return {
        1: _collect_ids(nodes, 1),
        2: _collect_ids(nodes, 2),
        3: _collect_ids(nodes, 3),
    }


def validate_classification(result: ClassificationResult, valid_ids: dict[int, set[str]]) -> list[str]:
    errors: list[str] = []
    if result.l1_id not in valid_ids[1]:
        errors.append(f"Unknown L1 id: {result.l1_id}")
    if result.l2_id not in valid_ids[2]:
        errors.append(f"Unknown L2 id: {result.l2_id}")
    if result.l3_id not in valid_ids[3]:
        errors.append(f"Unknown L3 id: {result.l3_id}")
    return errors


def check_hierarchy_consistency(result: ClassificationResult, taxonomy: dict) -> bool:
    """Verify that l3 is a descendant of l2, which is a descendant of l1."""
    def find_node(nodes: list[dict], target_id: str) -> dict | None:
        for node in nodes:
            if node.get("id") == target_id:
                return node
            for child in node.get("children", []):
                if isinstance(child, dict):
                    found = find_node([child], target_id)
                    if found:
                        return found
        return None

    l1_node = find_node(taxonomy["nodes"], result.l1_id)
    if not l1_node:
        return False
    l2_node = find_node(l1_node.get("children", []), result.l2_id)
    if not l2_node:
        return False
    l3_node = find_node(l2_node.get("children", []), result.l3_id)
    return l3_node is not None
