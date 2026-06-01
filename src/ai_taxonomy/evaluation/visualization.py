from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from .metrics import load_all_results, confidence_distribution, inter_model_agreement
from .. import config

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_confidence_distributions(all_results: dict, output_dir: Path = config.FIGURES_DIR) -> None:
    df = confidence_distribution(all_results)
    fig, ax = plt.subplots(figsize=(9, 4))
    for model, group in df.groupby("model"):
        ax.hist(group["confidence"], bins=20, alpha=0.6, label=model, density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Score Distribution by Model")
    ax.legend()
    _save(fig, "confidence_distribution.png", output_dir)


def plot_agreement_by_l1(all_results: dict, taxonomy: dict, output_dir: Path = config.FIGURES_DIR) -> None:
    l1_nodes = {n["id"]: n["label"] for n in taxonomy["nodes"]}
    domains = list(l1_nodes.keys())
    tool_ids = sorted(set.intersection(*[set(r.keys()) for r in all_results.values()]))

    rows = []
    for tid in tool_ids:
        l1_vals = [all_results[m][tid]["l1_id"] for m in all_results]
        dominant_l1 = max(set(l1_vals), key=l1_vals.count)
        agrees = len(set(l1_vals)) == 1
        rows.append({"l1_id": dominant_l1, "agreement": int(agrees)})

    df = pd.DataFrame(rows)
    agg = df.groupby("l1_id")["agreement"].mean().rename(index=l1_nodes).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    agg.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(agg)))
    ax.set_xlabel("Agreement Rate")
    ax.set_title("Inter-Model Agreement Rate by L1 Domain")
    ax.set_xlim(0, 1)
    _save(fig, "agreement_by_domain.png", output_dir)


def plot_l2_confusion_heatmap(all_results: dict, output_dir: Path = config.FIGURES_DIR) -> None:
    models = list(all_results.keys())
    if len(models) < 2:
        return
    m1, m2 = models[0], models[1]
    tool_ids = sorted(set(all_results[m1]) & set(all_results[m2]))

    labels1 = [all_results[m1][tid]["l2_label"] for tid in tool_ids]
    labels2 = [all_results[m2][tid]["l2_label"] for tid in tool_ids]

    all_labels = sorted(set(labels1) | set(labels2))
    matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)
    for l1, l2 in zip(labels1, labels2):
        matrix.loc[l1, l2] += 1

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(matrix, ax=ax, cmap="Blues", linewidths=0.3, annot=True, fmt="d", annot_kws={"size": 7})
    ax.set_title(f"L2 Classification Confusion: {m1} vs {m2}")
    ax.set_xlabel(m2)
    ax.set_ylabel(m1)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    _save(fig, "l2_confusion_heatmap.png", output_dir)


def plot_confidence_vs_agreement(all_results: dict, output_dir: Path = config.FIGURES_DIR) -> None:
    models = list(all_results.keys())
    tool_ids = sorted(set.intersection(*[set(r.keys()) for r in all_results.values()]))

    rows = []
    for tid in tool_ids:
        l3_vals = [all_results[m][tid]["l3_id"] for m in models]
        agrees = int(len(set(l3_vals)) == 1)
        avg_conf = sum(all_results[m][tid].get("confidence", 0) for m in models) / len(models)
        rows.append({"avg_confidence": avg_conf, "all_agree": agrees})

    df = pd.DataFrame(rows)
    bins = pd.cut(df["avg_confidence"], bins=10)
    agg = df.groupby(bins, observed=True)["all_agree"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    agg.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[0])
    ax.set_xlabel("Average Confidence Bucket")
    ax.set_ylabel("All-Model Agreement Rate")
    ax.set_title("Confidence vs Inter-Model Agreement")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _save(fig, "confidence_vs_agreement.png", output_dir)


def generate_all_figures(taxonomy: dict, output_dir: Path = config.FIGURES_DIR) -> None:
    all_results = load_all_results()
    if not all_results:
        print("No results found — skipping visualisations.")
        return
    plot_confidence_distributions(all_results, output_dir)
    plot_agreement_by_l1(all_results, taxonomy, output_dir)
    plot_l2_confusion_heatmap(all_results, output_dir)
    plot_confidence_vs_agreement(all_results, output_dir)
    print("All figures generated.")
