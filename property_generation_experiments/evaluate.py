"""Evaluate every model in coded_generations/ and write evaluations.md.

For each experiment with coded outputs, this script:
  1. Discovers every model directory under coded_generations/exp{1,2}/.
  2. Loads all run CSVs per model and averages per-word frequency vectors.
  3. Merges against data/{harpaintner,kelly}/human_norms_proper.csv and
     computes per-category Pearson r and the headline Mean r.
  4. Writes a markdown leaderboard to evaluations.md (one section per
     experiment, sorted by Mean r descending).

Usage:
    python evaluate.py
"""

import ast
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import ROOT, ExperimentConfig, get_config

OUT_PATH = os.path.join(ROOT, "evaluations.md")


def parse_freq(value) -> Optional[List[float]]:
    if isinstance(value, list):
        return [float(x) for x in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (ValueError, SyntaxError):
        return None
    return None


def load_run(path: str, cats: List[str]) -> Optional[pd.DataFrame]:
    df = pd.read_csv(path)
    if "word" not in df.columns or "frequencies" not in df.columns:
        return None
    df["word"] = df["word"].astype(str).str.strip()
    parsed = df["frequencies"].apply(parse_freq)
    keep = parsed.apply(lambda v: v is not None and len(v) == len(cats))
    df = df[keep].copy()
    if df.empty:
        return None
    freq_df = pd.DataFrame(parsed[keep].tolist(), columns=cats)
    return pd.concat([df["word"].reset_index(drop=True), freq_df], axis=1)


def discover_models(coded_dir: str) -> Dict[str, List[str]]:
    """Map model_prefix -> list of run CSV paths."""
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(coded_dir):
        return out
    for entry in sorted(os.listdir(coded_dir)):
        sub = os.path.join(coded_dir, entry)
        if not os.path.isdir(sub):
            continue
        runs = sorted(glob.glob(os.path.join(sub, "*_coded_with_*_run_*.csv")))
        if runs:
            out[entry] = runs
    return out


def aggregate(runs_paths: List[str], cats: List[str]) -> Optional[pd.DataFrame]:
    frames = [load_run(p, cats) for p in runs_paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("word", as_index=False)[cats].mean()


def correlate(model_df: pd.DataFrame, human_df: pd.DataFrame, word_col: str,
              cats: List[str]) -> Optional[Dict[str, float]]:
    merged = model_df.merge(human_df, left_on="word", right_on=word_col, suffixes=("_m", "_h"))
    if len(merged) < 10:
        return None
    rs = {}
    for cat in cats:
        m_col = f"{cat}_m" if f"{cat}_m" in merged.columns else cat + "_m"
        h_col = f"{cat}_h" if f"{cat}_h" in merged.columns else cat + "_h"
        m_vals = merged[m_col].astype(float).values
        h_vals = merged[h_col].astype(float).values
        r, _ = stats.pearsonr(m_vals, h_vals)
        rs[cat] = float(r)
    rs["mean"] = float(np.mean(list(rs.values())))
    rs["n"] = len(merged)
    return rs


def evaluate_experiment(cfg: ExperimentConfig) -> Tuple[List[Dict], Optional[float]]:
    human = cfg.load_norms()
    human[cfg.word_column] = human[cfg.word_column].astype(str).str.strip()

    rows = []
    models = discover_models(cfg.coded_dir)
    for prefix, run_paths in models.items():
        agg = aggregate(run_paths, cfg.cats)
        if agg is None:
            continue
        rs = correlate(agg, human, cfg.word_column, cfg.cats)
        if rs is None:
            continue
        rs["model"] = prefix
        rs["runs"] = len(run_paths)
        rows.append(rs)

    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows, None


def render_table(cfg: ExperimentConfig, rows: List[Dict]) -> str:
    if not rows:
        return f"No coded runs found under `coded_generations/{cfg.name}/`.\n"
    header = ["Rank", "Model", "Runs", "Mean r"] + cfg.cat_labels
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for i, r in enumerate(rows, 1):
        cells = [str(i), f"`{r['model']}`", str(r["runs"]), f"{r['mean']:.3f}"]
        cells += [f"{r[c]:.3f}" for c in cfg.cats]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    chunks = ["# Evaluations", ""]
    for exp_id in (1, 2):
        cfg = get_config(exp_id)
        rows, _ = evaluate_experiment(cfg)
        title = "Experiment 1 (Harpaintner et al. 2018)" if exp_id == 1 else "Experiment 2 (Kelly et al. 2024)"
        chunks.append(f"## {title}")
        chunks.append("")
        chunks.append(render_table(cfg, rows))
        chunks.append("")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(chunks))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
