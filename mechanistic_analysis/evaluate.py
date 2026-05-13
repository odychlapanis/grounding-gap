"""Per-dimension Pearson correlations between SAE feature activations and the
Troche human ratings.

For each variant_modeltag dir under `activations/` and each of the 14 Troche
dimensions, this script:
  1. Loads `activations/<variant>_<model>/prompt{1,2,3}.json`.
  2. Computes Pearson r per (layer, feature_id, dim, prompt) on the 751-word
     pool, dropping zero-variance features.
  3. Drops features with negative r on any of the 3 prompts (a sign-flip
     across phrasings means the binding is prompt-specific, so the mean is
     not meaningful).
  4. Averages the remaining (layer, feature_id, dim) Pearson r values across
     prompts.
  5. Writes one markdown table per dimension to
     `evaluations/<variant>_<model>/<dim>.md`, listing the top 100 features
     by mean Pearson r.

Usage (from mechanistic_analysis/):
    python evaluate.py
    python evaluate.py --variant resid_post_4b
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ACT_ROOT = os.path.join(HERE, "activations")
EVAL_ROOT = os.path.join(HERE, "evaluations")
HUMAN_NORMS = os.path.abspath(os.path.join(
    HERE, "..", "rating_experiment", "data", "human_ground_truth_ratings.csv"))

DIMENSIONS = [
    "emotion", "polarity", "social", "moral", "motion_self", "thought",
    "color", "taste_smell", "tactile", "visual_form", "auditory",
    "space", "quantity", "time",
]
TOP_N = 100


def normalise(w: str) -> str:
    return str(w).strip().lower()


def parse_layer(s: str) -> int:
    return int(s.split("_")[-1] if "layer" in s else s)


def discover_variants() -> list[str]:
    if not os.path.isdir(ACT_ROOT):
        return []
    return sorted(d for d in os.listdir(ACT_ROOT)
                  if os.path.isdir(os.path.join(ACT_ROOT, d)))


def load_prompt_acts(act_dir: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in (1, 2, 3):
        path = os.path.join(act_dir, f"prompt{p}.json")
        if not os.path.exists(path):
            continue
        out[p] = {normalise(w): v for w, v in json.load(open(path)).items()}
    return out


def feature_matrix(acts: dict, words: list[str]) -> tuple[dict, np.ndarray]:
    """Index each (layer, feat_id) -> dense vector over `words`. Zero if absent."""
    n = len(words)
    word2idx = {w: i for i, w in enumerate(words)}
    feats: dict[tuple[int, int], np.ndarray] = {}
    for w, ldict in acts.items():
        wi = word2idx.get(w)
        if wi is None:
            continue
        for layer_str, fdict in ldict.items():
            try:
                L = parse_layer(layer_str)
            except ValueError:
                continue
            for fid, v in fdict.items():
                k = (L, int(fid))
                if k not in feats:
                    feats[k] = np.zeros(n, dtype=np.float32)
                feats[k][wi] = v
    return feats, None


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    r, _ = stats.pearsonr(x, y)
    return float(r) if np.isfinite(r) else float("nan")


def evaluate_variant(variant_tag: str, human: pd.DataFrame) -> None:
    act_dir = os.path.join(ACT_ROOT, variant_tag)
    prompts = load_prompt_acts(act_dir)
    if not prompts:
        print(f"  {variant_tag}: no prompt JSONs found, skipping")
        return
    print(f"\n[{variant_tag}] prompts available: {sorted(prompts)}")

    words = sorted(set.intersection(*[set(p.keys()) for p in prompts.values()])
                   & set(human.index))
    print(f"  words usable across all prompts: {len(words)}")

    rating_mat = np.full((len(words), len(DIMENSIONS)), np.nan)
    for i, w in enumerate(words):
        row = human.loc[w]
        for j, d in enumerate(DIMENSIONS):
            v = row[d]
            if pd.notna(v):
                rating_mat[i, j] = float(v)

    per_prompt_r: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    feat_keys_by_prompt: dict[int, set] = {}
    for p, acts in prompts.items():
        feats, _ = feature_matrix(acts, words)
        feat_keys_by_prompt[p] = set(feats.keys())
        rmat = np.full((len(feats), len(DIMENSIONS)), np.nan, dtype=np.float64)
        keys = list(feats.keys())
        for fi, k in enumerate(keys):
            vec = feats[k]
            for di in range(len(DIMENSIONS)):
                ratings = rating_mat[:, di]
                mask = ~np.isnan(ratings)
                if mask.sum() < 3:
                    continue
                rmat[fi, di] = pearson(vec[mask].astype(np.float64), ratings[mask])
        per_prompt_r[p] = {k: rmat[i] for i, k in enumerate(keys)}
        print(f"  prompt{p}: {len(feats)} unique (layer, feature) pairs scored")

    common_keys = sorted(set.intersection(*feat_keys_by_prompt.values()))
    print(f"  features common to all prompts: {len(common_keys)}")

    out_dir = os.path.join(EVAL_ROOT, variant_tag)
    os.makedirs(out_dir, exist_ok=True)

    for di, dim in enumerate(DIMENSIONS):
        rows = []
        for k in common_keys:
            rs = [per_prompt_r[p][k][di] for p in sorted(per_prompt_r)]
            if any(np.isnan(r) for r in rs):
                continue
            if any(r <= 0 for r in rs):
                continue
            rows.append((k[0], k[1], float(np.mean(rs)), rs))
        rows.sort(key=lambda r: -r[2])
        rows = rows[:TOP_N]

        lines = [
            f"# {dim} -- top {len(rows)} SAE features ({variant_tag})",
            "",
            f"Pearson r between SAE feature activation and Troche `{dim}` ratings,",
            f"averaged across {len(per_prompt_r)} prompt phrasings on the {len(words)}-word pool.",
            "Features with negative r on any prompt are excluded.",
            "",
            "| Rank | Layer | Feature ID | Mean r | r prompt1 | r prompt2 | r prompt3 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for i, (L, fid, mean_r, rs) in enumerate(rows, 1):
            cells = [str(i), str(L), str(fid), f"{mean_r:.4f}"]
            for p in (1, 2, 3):
                if p in per_prompt_r:
                    cells.append(f"{rs[sorted(per_prompt_r).index(p)]:.4f}")
                else:
                    cells.append("--")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        out_path = os.path.join(out_dir, f"{dim}.md")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
    print(f"  wrote {len(DIMENSIONS)} markdown files under {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None,
                    help="restrict to one activations subdir (default: all under activations/)")
    args = ap.parse_args()

    if not os.path.exists(HUMAN_NORMS):
        raise SystemExit(f"missing human ratings: {HUMAN_NORMS}")
    human = pd.read_csv(HUMAN_NORMS)
    human["word"] = human["word"].map(normalise)
    human = human.drop_duplicates(subset=["word"]).set_index("word")

    variants = [args.variant] if args.variant else discover_variants()
    if not variants:
        raise SystemExit(f"no activation directories under {ACT_ROOT}/")
    for v in variants:
        evaluate_variant(v, human)


if __name__ == "__main__":
    main()
