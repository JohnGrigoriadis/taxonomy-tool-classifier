"""
Download and process AI tool data from multiple sources, then save to data/processed/tools.jsonl.

Sources:
  1. HuggingFace dataset  maharshipandya/huggingface-tools
  2. HuggingFace dataset  fka/awesome-chatgpt-prompts
  3. RankmyAI rankings page (scraped, robots.txt-respecting, 1 req/s)
"""
import json
import time
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_taxonomy import config
from ai_taxonomy.data.preprocessor import preprocess


def _fetch_hf_tools() -> list[dict]:
    ds = load_dataset("maharshipandya/huggingface-tools", split="train")
    tools = []
    for i, row in enumerate(ds):
        tools.append({
            "id": f"hf-tools-{i:04d}",
            "name": row.get("name") or row.get("tool_name") or f"tool-{i}",
            "description": row.get("description") or "",
            "source": "huggingface-tools",
        })
    return tools


def _fetch_chatgpt_prompts() -> list[dict]:
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    tools = []
    for i, row in enumerate(ds):
        tools.append({
            "id": f"cgpt-{i:04d}",
            "name": row.get("act") or f"prompt-{i}",
            "description": row.get("prompt") or "",
            "source": "awesome-chatgpt-prompts",
        })
    return tools


def _fetch_rankmyai(max_pages: int = 5) -> list[dict]:
    base_url = "https://www.rankmyai.com/rankings"
    headers = {"User-Agent": "taxonomy-research-bot/1.0 (academic; non-commercial)"}
    tools: list[dict] = []

    try:
        robots = httpx.get("https://www.rankmyai.com/robots.txt", timeout=10, headers=headers)
        if "Disallow: /rankings" in robots.text:
            print("robots.txt disallows /rankings — skipping RankmyAI scrape.")
            return tools
    except Exception:
        pass

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        try:
            r = httpx.get(url, timeout=15, headers=headers)
            r.raise_for_status()
        except Exception as e:
            print(f"RankmyAI page {page} failed: {e}")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select("[data-tool-name], .tool-card, article.tool")
        if not cards:
            break

        for card in cards:
            name = (
                card.get("data-tool-name")
                or (card.select_one("h2, h3, .tool-name") or {}).get_text(strip=True)
                or ""
            )
            desc_el = card.select_one("p, .description, .tool-description")
            description = desc_el.get_text(strip=True) if desc_el else ""
            if name:
                tools.append({
                    "id": f"rankmyai-{len(tools):04d}",
                    "name": name,
                    "description": description,
                    "source": "rankmyai",
                })

        time.sleep(1.0)

    return tools


def main() -> None:
    print("Fetching HuggingFace tools dataset...")
    raw: list[dict] = []
    raw.extend(_fetch_hf_tools())
    print(f"  → {len(raw)} tools")

    print("Fetching awesome-chatgpt-prompts...")
    prompts = _fetch_chatgpt_prompts()
    raw.extend(prompts)
    print(f"  → {len(prompts)} items")

    print("Scraping RankmyAI rankings...")
    rankmyai = _fetch_rankmyai()
    raw.extend(rankmyai)
    print(f"  → {len(rankmyai)} tools")

    print(f"\nTotal raw: {len(raw)}")
    tools = preprocess(raw)
    print(f"After dedup + filter: {len(tools)}")

    out_path = config.TOOLS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for tool in tools:
            f.write(json.dumps(tool) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
