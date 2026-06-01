"""
One-time script: uses Claude to propose L3 leaf nodes for each L1+L2 pair, then
writes the result to data/taxonomy.json for manual review and editing.

Run:  uv run python scripts/build_taxonomy.py
"""
import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_taxonomy import config

_PROMPT = """You are designing a hierarchical taxonomy for AI tools.

Below is the current taxonomy with L1 (domain) and L2 (capability) nodes.
Your task: propose 2–5 L3 (technique) leaf nodes for each L2 node.

Requirements:
- L3 labels should be concrete, mutually exclusive techniques/use-cases within the L2.
- Each label ≤ 5 words, title case.
- Return ONLY a JSON array where each element has:
  {{
    "l2_id": "<id>",
    "l3_nodes": [
      {{"id": "<new-L3-id>", "level": 3, "label": "<label>"}}
    ]
  }}

Current taxonomy (L1 and L2 only):
{skeleton}
"""


def _build_skeleton(taxonomy: dict) -> str:
    lines = []
    for l1 in taxonomy["nodes"]:
        lines.append(f"L1 {l1['id']}: {l1['label']}")
        for l2 in l1.get("children", []):
            if isinstance(l2, dict):
                lines.append(f"  L2 {l2['id']}: {l2['label']} — {l2.get('description', '')}")
    return "\n".join(lines)


def main() -> None:
    taxonomy_path = config.TAXONOMY_PATH
    if not taxonomy_path.exists():
        print(f"taxonomy.json not found at {taxonomy_path}. Create a skeleton first.")
        sys.exit(1)

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    skeleton = _build_skeleton(taxonomy)
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    print("Requesting L3 proposals from Claude...")
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": _PROMPT.format(skeleton=skeleton)}],
    )
    raw = response.content[0].text
    proposals = json.loads(raw)

    l2_map: dict[str, dict] = {}
    for l1 in taxonomy["nodes"]:
        for l2 in l1.get("children", []):
            if isinstance(l2, dict):
                l2_map[l2["id"]] = l2

    for item in proposals:
        l2 = l2_map.get(item["l2_id"])
        if l2 is not None:
            existing = {n["id"] for n in l2.get("children", []) if isinstance(n, dict)}
            for node in item["l3_nodes"]:
                if node["id"] not in existing:
                    l2.setdefault("children", []).append(node)

    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"taxonomy.json updated at {taxonomy_path}.")
    print("Review and edit the file before running classification.")


if __name__ == "__main__":
    main()
