import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from .. import config


def _load_results(path: Path) -> dict[str, dict]:
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                results[r["tool_id"]] = r
    return results


def load_all_results() -> dict[str, dict[str, dict]]:
    """Returns {model_name: {tool_id: result}}"""
    all_results: dict[str, dict[str, dict]] = {}
    for path in sorted(config.CLASSIFICATIONS_DIR.glob("*.jsonl")):
        model_name = path.stem.replace("_results", "")
        all_results[model_name] = _load_results(path)
    return all_results


def inter_model_agreement(all_results: dict[str, dict[str, dict]], level: str = "l1_id") -> float:
    """% of tools where all models agree on the given level."""
    tool_ids = set.intersection(*[set(r.keys()) for r in all_results.values()])
    if not tool_ids:
        return 0.0
    agree = sum(
        len({r[tid][level] for r in all_results.values()}) == 1
        for tid in tool_ids
    )
    return agree / len(tool_ids)


def pairwise_kappa(all_results: dict[str, dict[str, dict]], level: str = "l3_id") -> pd.DataFrame:
    models = list(all_results.keys())
    tool_ids = sorted(set.intersection(*[set(r.keys()) for r in all_results.values()]))
    rows = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            labels1 = [all_results[m1][tid][level] for tid in tool_ids]
            labels2 = [all_results[m2][tid][level] for tid in tool_ids]
            kappa = cohen_kappa_score(labels1, labels2)
            rows.append({"model_a": m1, "model_b": m2, "kappa": round(kappa, 4)})
    return pd.DataFrame(rows)


def ambiguity_rate(all_results: dict[str, dict[str, dict]]) -> pd.Series:
    rates = {}
    for model, results in all_results.items():
        flagged = sum(1 for r in results.values() if r.get("ambiguous", False))
        rates[model] = flagged / len(results) if results else 0.0
    return pd.Series(rates, name="ambiguity_rate")


def confidence_distribution(all_results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for model, results in all_results.items():
        for r in results.values():
            rows.append({"model": model, "confidence": r.get("confidence", 0.0)})
    return pd.DataFrame(rows)


def run_all_metrics(output_dir: Path = config.EVALUATION_DIR) -> None:
    all_results = load_all_results()
    if not all_results:
        print("No classification results found.")
        return

    print(f"\n{'='*50}")
    print(f"Models evaluated: {list(all_results.keys())}")

    l1_agree = inter_model_agreement(all_results, "l1_id")
    l3_agree = inter_model_agreement(all_results, "l3_id")
    print(f"\nInter-model agreement @ L1: {l1_agree:.1%}")
    print(f"Inter-model agreement @ L3: {l3_agree:.1%}")

    kappa_df = pairwise_kappa(all_results)
    print(f"\nPairwise Cohen's κ (L3):\n{kappa_df.to_string(index=False)}")
    kappa_df.to_csv(output_dir / "inter_model_kappa.csv", index=False)

    ambig = ambiguity_rate(all_results)
    print(f"\nAmbiguity rate per model:\n{ambig.to_string()}")

    conf_df = confidence_distribution(all_results)
    conf_df.to_csv(output_dir / "confidence_distribution.csv", index=False)

    agree_df = pd.DataFrame({
        "level": ["L1", "L3"],
        "agreement": [l1_agree, l3_agree],
    })
    agree_df.to_csv(output_dir / "inter_model_agreement.csv", index=False)
    print(f"\nMetrics saved to {output_dir}")
