"""Evaluate every model in ratings/ and write evaluations.md.

For each model directory under `ratings/`, this script:
  1. Averages per-word ratings across seeds for each of the 14 dimensions.
  2. Computes Pearson r against the human norms in
     `data/human_ground_truth_ratings.csv`.
  3. Reports component means using the Troche (2017) groupings:
       Internal:  emotion, polarity, social, moral, motion_self, thought
       Sensory:   visual_form, tactile, taste_smell, auditory, color
       Magnitude: space, quantity, time
  4. Computes Mean r as the mean across all 14 dimensions.
  5. Writes a markdown leaderboard to `evaluations.md`, sorted by Mean r descending.

Usage (run from the rating_experiment/ directory):
    python src/evaluate.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_NORMS = os.path.join(ROOT, "data", "human_ground_truth_ratings.csv")
RATINGS_ROOT = os.path.join(ROOT, "ratings")
OUT_PATH = os.path.join(ROOT, "evaluations.md")

DIMENSIONS = [
    "emotion", "polarity", "social", "moral", "motion_self", "thought",
    "color", "taste_smell", "tactile", "visual_form", "auditory",
    "space", "quantity", "time",
]

COMPONENTS = {
    "Internal":  ["emotion", "polarity", "social", "moral", "motion_self", "thought"],
    "Sensory":   ["color", "taste_smell", "tactile", "visual_form", "auditory"],
    "Magnitude": ["space", "quantity", "time"],
}


def aggregate_seeds(dim_dir: str) -> pd.DataFrame:
    seed_files = sorted(glob.glob(os.path.join(dim_dir, "seed_*.csv")))
    if not seed_files:
        return pd.DataFrame(columns=["word", "rating"])
    frames = [pd.read_csv(p) for p in seed_files]
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["rating"])
    combined["word"] = combined["word"].astype(str).str.strip()
    return combined.groupby("word", as_index=False)["rating"].mean()


def pearson(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return float("nan")
    r, _ = stats.pearsonr(a[mask], b[mask])
    return float(r)


def evaluate_model(model_dir: str, human: pd.DataFrame) -> dict:
    row = {"model": os.path.basename(model_dir)}
    dim_rs = {}
    for dim in DIMENSIONS:
        dim_dir = os.path.join(model_dir, dim)
        if not os.path.isdir(dim_dir):
            dim_rs[dim] = float("nan")
            continue
        agg = aggregate_seeds(dim_dir)
        if agg.empty:
            dim_rs[dim] = float("nan")
            continue
        merged = human[["word", dim]].merge(agg, on="word", how="inner")
        dim_rs[dim] = pearson(merged[dim], merged["rating"])

    for component, dims in COMPONENTS.items():
        row[component] = float(np.nanmean([dim_rs[d] for d in dims]))
    row["Mean r"] = float(np.nanmean([dim_rs[d] for d in DIMENSIONS]))
    for dim in DIMENSIONS:
        row[dim] = dim_rs[dim]
    row["seeds"] = _count_seeds(model_dir)
    return row


def _count_seeds(model_dir: str) -> int:
    counts = []
    for dim in DIMENSIONS:
        dim_dir = os.path.join(model_dir, dim)
        if os.path.isdir(dim_dir):
            counts.append(len(glob.glob(os.path.join(dim_dir, "seed_*.csv"))))
    return max(counts) if counts else 0


def discover_models() -> list[str]:
    if not os.path.isdir(RATINGS_ROOT):
        return []
    return sorted(
        os.path.join(RATINGS_ROOT, d)
        for d in os.listdir(RATINGS_ROOT)
        if os.path.isdir(os.path.join(RATINGS_ROOT, d))
    )


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "No model directories found under `ratings/`.\n"
    header = ["Rank", "Model", "Seeds", "Mean r", "Internal", "Sensory", "Magnitude"] + DIMENSIONS
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for i, r in enumerate(rows, 1):
        cells = [
            str(i), f"`{r['model']}`", str(r["seeds"]),
            f"{r['Mean r']:.3f}",
            f"{r['Internal']:.3f}", f"{r['Sensory']:.3f}", f"{r['Magnitude']:.3f}",
        ]
        cells += [f"{r[d]:.3f}" if not np.isnan(r[d]) else "—" for d in DIMENSIONS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    human = pd.read_csv(HUMAN_NORMS)
    human["word"] = human["word"].astype(str).str.strip()

    rows = []
    for model_dir in discover_models():
        rows.append(evaluate_model(model_dir, human))
    rows.sort(key=lambda r: (-r["Mean r"] if not np.isnan(r["Mean r"]) else 1.0))

    chunks = [
        "# Rating-experiment evaluations",
        "",
        "Pearson r between mean model ratings and Troche (2017) human norms,",
        "sorted by Mean r (mean across the 14 dimensions).",
        "",
        render_table(rows),
    ]
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(chunks))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
