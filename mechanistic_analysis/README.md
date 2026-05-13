# Mechanistic analysis

Code, configs, and a sample activation cache accompanying the paper

> **The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans**

This directory reproduces the SAE feature analysis: for every
ACF dimension, find which Gemma residual-stream features (decoded through the
GemmaScope SAEs) correlate best with the human ratings. Three prompt
phrasings are extracted and averaged so the surfaced features are stable
across surface phrasing and not artefacts of any single prompt.

## Pipeline

| Step | Script | Output |
| :--- | :--- | :--- |
| 1. Extract activations on the 751-word Troche pool, three prompts. | `extract_rating_activations.py` | `activations/<variant>_<model>/prompt{1,2,3}.json` |
| 2. Pearson r per (layer, feature, dimension), averaged across prompts. | `evaluate.py` | `evaluations/<variant>_<model>/<dim>.md` (top 100 features per dimension) |

Step 1 needs a CUDA GPU. Step 2 is CPU-only; reviewers can run it directly
on the shipped sample without any GPU.

## What is shipped

A sample activation cache for the smallest configuration is pre-computed so
the leaderboard step works out of the box:

- `activations/resid_post_4b/prompt{1,2,3}.json` -- Gemma 3 4B IT, GemmaScope
  16k_medium SAEs at layers 9, 17, 22, 29, last-token features over the 751
  Troche stimuli, for each of the three prompts.



## Try it: leaderboard from the shipped sample

No GPU and no API key required. From this directory:

```evaluate
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy tqdm
python evaluate.py --variant resid_post_4b
ls evaluations/resid_post_4b/
```

Each of the 14 dimension markdown files lists the top-100 SAE features by
mean Pearson r across the three prompts, with per-prompt r columns alongside.

## Reproduce or extend the activations (GPU)

The full GPU stack -- including SAELens and TransformerLens cloned at the
exact commits used for the paper -- is set up like this:

### 1. Python deps

```setup
python -m venv .venv-gpu          # or use a conda env
source .venv-gpu/bin/activate
pip install -r requirements.txt
```

### 2. Clone SAELens and TransformerLens at the paper-canonical commits

```libs
mkdir -p libs && cd libs
git clone https://github.com/jbloomAus/SAELens.git
( cd SAELens && git checkout 99e395ea3da0b9a4dccd54f7ce24c5646f285385 )
git clone https://github.com/TransformerLensOrg/TransformerLens.git
( cd TransformerLens && git checkout 7df72ff71b3b0b25845f9d12836ba45a58a0d629 )
cd ..
```

These clones are loaded via PYTHONPATH (no `pip install -e`). The two pinned
commits are the ones that produced the activation caches shipped here.

### 3. Run extraction

```extract
PYTHONPATH=libs/SAELens:libs/TransformerLens \
HF_TOKEN=<your hf token, gated Gemma weights> \
python extract_rating_activations.py --config configs/4b_resid_post.json
```

Defaults to all three prompts on the 751-word pool. Restrict with
`--prompts 1 2`, restrict layers with `--layers 9,17`, or cap words with
`--max_words 16` for a smoke test. Re-running the same command tops up
missing (word, layer, prompt) triples, so it is safe to interrupt.

Wall time on a single 24 GB CUDA GPU (bf16, no batching):

| Config | Layers | Words x layers x prompts | Wall time |
| :--- | :--- | :--- | :--- |
| `4b_resid_post` | 4 | 9 012 | ~10 min |
| `4b_resid_post_all` | 34 | 76 602 | ~85 min |
| `12b_resid_post` (n_devices=2) | 4 | 9 012 | ~20 min |
| `12b_resid_post_all` (n_devices=2) | 41 | 92 373 | ~6 h |

### 4. Build the leaderboards

```evaluate-gpu
python evaluate.py
```

Picks up every directory under `activations/`, builds the per-dimension
markdown files under `evaluations/<variant>_<model>/`.

## Layout

```
mechanistic_analysis/
  README.md                           this file
  requirements.txt                    GPU dependency set
  extract_rating_activations.py       step 1: GPU SAE feature extraction
  evaluate.py                         step 2: per-dimension Pearson r
  configs/
    4b_resid_post.json                Gemma 3 4B IT, 16k_medium, 4 layers
    4b_resid_post_all.json            Gemma 3 4B IT, 16k_big, all 34 layers
    12b_resid_post.json               Gemma 3 12B IT, 16k_medium, 4 layers
    12b_resid_post_all.json           Gemma 3 12B IT, 16k_big, all 41 layers
  data/
    prompts/                          three prompt phrasings (markdown)
  activations/
    resid_post_4b/prompt{1,2,3}.json  shipped sample activation cache
  evaluations/                        produced by evaluate.py (gitignored)
  libs/                               GPU clones of SAELens / TransformerLens
                                      (created in step 2 above; gitignored)
```

The Rating Experiment human ratings live in
`../rating_experiment/data/human_ground_truth_ratings.csv` and are read
directly from there by both scripts -- no copy is duplicated here.
