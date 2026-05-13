# Experiment 2 data (Kelly et al. 2024)

The Kelly et al. (2024) feature-norms dataset is **not bundled** with this
repository. It is hosted publicly by the original authors on the Open
Science Framework at <https://osf.io/eh5dk/> and is downloaded on demand.

## Setup

Run once, before `evaluate.py`:

```bash
python data/experiment_2/fetch_kelly.py
```

The script:

1. Downloads the Kelly source files into `raw/` (preserved as-is).
2. Builds `coding_human_norms_ground_truth.csv`: the per-clue table filtered
   to the Abstract + Emotion conditions (used as gold for evaluating the
   coding prompt).
3. Builds `human_norms.csv`: per-concept proportions across the four
   substantive categories (taxonomic, entity, situation, introspective),
   used as gold for Experiment 2 scoring.

Re-run with `--force` to refresh the downloads.

## Files in this directory

| File | Source | License |
| :--- | :--- | :--- |
| `generation_prompt.md` | this repository | Apache-2.0 (see `../../../LICENSE`) |
| `coding_prompt.md` | this repository | Apache-2.0 |
| `fetch_kelly.py` | this repository | Apache-2.0 |
| `raw/*` (after fetch) | OSF eh5dk | Kelly et al. 2024, no explicit license |
| `human_norms.csv` (after fetch) | derived from OSF data | same as source |
| `coding_human_norms_ground_truth.csv` (after fetch) | derived from OSF data | same as source |

Please cite:

> Kelly, A. E., et al. (2024). Conceptual Structure of Emotions. *Emotion*.
> Data: <https://osf.io/eh5dk/>

See `../../../DATA_LICENSES.md` for the full dataset-licensing notice.
