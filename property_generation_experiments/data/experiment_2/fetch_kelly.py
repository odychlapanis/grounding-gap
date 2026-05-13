"""Fetch Kelly et al. (2024) data from OSF and build the derived norms.

This repository does NOT bundle the Kelly et al. (2024) emotion-concepts
dataset. It is hosted publicly by the original authors at
https://osf.io/eh5dk/ and is downloaded on demand by this script.

The script performs three steps:
  1. Download the source files from OSF into ``raw/`` (preserved as-is).
  2. Build ``coding_human_norms_ground_truth.csv``: the per-clue table
     ``coded_features.csv`` filtered to the Abstract + Emotion conditions
     (all clues, every labelMajor). Used as the per-clue gold standard for
     evaluating the coding prompt.
  3. Build ``human_norms.csv``: one row per concept with per-category
     proportions (taxonomic, entity, situation, introspective), aggregated
     from the same filtered table after dropping the residual ``misc``
     label. This is the file scored in Experiment 2.

Citation:
  Kelly, A. E., et al. (2024). Conceptual Structure of Emotions. Emotion.
  Data: https://osf.io/eh5dk/

Reuse note: the OSF project is public but carries no explicit license. Each
user pulls from the canonical author-published source for academic use; this
repository does not redistribute the data. Contact the authors for any
further redistribution.

Usage:
  python fetch_kelly.py            # download + build
  python fetch_kelly.py --force    # re-download even if files exist
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

OSF_PROJECT = "https://osf.io/eh5dk/"

SOURCE_FILES = {
    "coded_features.csv":   "https://osf.io/download/zuj8x/",
    "all_raw_features.csv": "https://osf.io/download/amq73/",
    "Data_README.txt":      "https://osf.io/download/eydnm/",
    "Stimuli.csv":          "https://osf.io/download/r9phv/",
    "Stimuli_README.txt":   "https://osf.io/download/meb47/",
}

CATEGORIES = ["taxonomic", "entity", "situation", "introspective"]
KEEP_CONDITIONS = {"Abstract", "Emotion"}
DROP_LABEL = "misc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
    tmp.replace(dest)


def build_coding_ground_truth(coded_csv: Path, out_csv: Path) -> int:
    """Filter coded_features.csv to the Abstract + Emotion conditions."""
    n = 0
    with coded_csv.open(newline="", encoding="utf-8") as f_in, \
         out_csv.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(
            f_out, fieldnames=reader.fieldnames, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for row in reader:
            if row["condition"] in KEEP_CONDITIONS:
                writer.writerow(row)
                n += 1
    return n


def build_human_norms(ground_truth_csv: Path, out_csv: Path) -> int:
    """Aggregate per-clue rows into per-concept category proportions."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with ground_truth_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["labelMajor"]
            if label == DROP_LABEL or label not in CATEGORIES:
                continue
            counts[row["concept"].strip().lower()][label] += 1

    rows = []
    for concept in sorted(counts):
        c = counts[concept]
        total = sum(c.get(k, 0) for k in CATEGORIES)
        if total == 0:
            continue
        rows.append([concept] + [f"{c.get(k, 0) / total:.6g}" for k in CATEGORIES])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept"] + CATEGORIES)
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files exist")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    raw_dir = here / "raw"
    ground_truth = here / "coding_human_norms_ground_truth.csv"
    norms = here / "human_norms.csv"

    print(f"Fetching Kelly et al. (2024) data from {OSF_PROJECT}")
    print(f"Raw dir: {raw_dir}\n")

    for name, url in SOURCE_FILES.items():
        out = raw_dir / name
        if out.exists() and not args.force:
            print(f"  [skip] {name} ({out.stat().st_size:,} bytes)")
            continue
        print(f"  [get ] {name}")
        try:
            download(url, out)
        except Exception as e:
            print(f"  [fail] {name}: {e}", file=sys.stderr)
            return 1
        print(f"         {out.stat().st_size:,} bytes  sha256={sha256(out)[:16]}...")

    print(f"\nBuilding {ground_truth.name} from coded_features.csv ...")
    n_clues = build_coding_ground_truth(raw_dir / "coded_features.csv", ground_truth)
    print(f"  wrote {ground_truth} ({n_clues:,} rows)")

    print(f"Building {norms.name} from {ground_truth.name} ...")
    n_concepts = build_human_norms(ground_truth, norms)
    print(f"  wrote {norms} ({n_concepts} concepts)")

    print(
        "\nDone. Kelly originals preserved in raw/;\n"
        "derived CSVs written next to this script and ready for evaluate.py.\n\n"
        "Please cite:\n"
        "  Kelly, A. E., et al. (2024). Conceptual Structure of Emotions. Emotion.\n"
        f"  Data: {OSF_PROJECT}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
