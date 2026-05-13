# Property generation experiments

Code, data, and sample runs accompanying the paper

> **The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans**

This directory reproduces Experiments 1 and 2 of the paper. Both experiments prompt an LLM for free property generation on a fixed stimulus list, code each generated property with an LLM coder, and score the per-stimulus category-frequency profile against the published human norms via Pearson r.

| | Experiment 1 | Experiment 2 |
| :--- | :--- | :--- |
| stimuli | 293 abstract nouns (Harpaintner et al. 2018) | 235 concepts (Abstract + Emotion subset of Kelly et al. 2024) |
| categories | Sensorimotor, Internal, Social, Verbal | Taxonomic, Entity, Situation, Introspective |
| canonical coder | `google/gemini-2.5-flash-lite` | `google/gemini-2.5-flash` |

## Requirements

The same environment serves all experiments in `supplementary/`. From the
`supplementary/` root:

```setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
$EDITOR .env  # add your OPENROUTER_API_KEY
```

The last three lines of `requirements.txt` (`torch`, `transformers`, `accelerate`) are commented out by default. Uncomment them only if you want to run open-weight models on a local GPU via `--client local`; the OpenRouter path does not need them.

All API calls (generation and coding) go through OpenRouter using the OpenAI Python SDK, so any OpenRouter-supported model id works (e.g. `openai/gpt-5.4`, `anthropic/claude-opus-4.6`, `google/gemini-2.5-flash`).

## One-time setup for Experiment 2

The Kelly et al. (2024) data is not bundled (see `data/experiment_2/README.md`). Run once before evaluating Experiment 2:
```fetch
python data/experiment_2/fetch_kelly.py
```
This downloads the source files from the authors' OSF project (<https://osf.io/eh5dk/>) and builds the per-clue and per-concept gold-reference CSVs.

## Try it: see the leaderboard on the preloaded sample

The repository ships with 10 generation runs and 10 coded runs for five sample models (`gemma-3-4b-it`, `gemma-3-12b-it`, `claude-opus-4.6`, `gemini-3.1-pro`, `openai-gpt-5.4`) on both experiments. Reproduce the headline leaderboards immediately, no API key required:
```evaluate
python evaluate.py
cat evaluations.md
```

## Reproduce or extend with a new model

The pipeline has three idempotent steps. Run them in order on any OpenRouter or HuggingFace model id.

### 1. Generate properties
```generate
python generate.py --experiment 1 --model openai/gpt-5.4
python generate.py --experiment 2 --model anthropic/claude-sonnet-4-6 --runs 10
python generate.py --experiment 1 --model google/gemma-3-4b-it --client local
```
Each run writes `generations/exp{1,2}/<safe_model>_run_<N>.csv` with columns `word, response, properties`. Re-running with the same `--model` resumes from the highest existing run number; within a run, already-processed words are skipped, so it is safe to interrupt.

Pass `--name <prefix>` to override the default safe-model filename prefix (useful when you want a new run to land alongside existing files that use a different naming convention).

### 2. Code generated properties
```code
python code.py --experiment 1 --model openai/gpt-5.4
python code.py --experiment 2 --model anthropic/claude-sonnet-4-6
```
Each invocation codes every available run for the given model with the canonical coder. Override with `--coder <model_id>`. Output goes to `coded_generations/exp{1,2}/<safe_model>/<safe_model>_coded_with_<safe_coder>_run_<N>.csv` with columns `word, properties, codes, frequencies`. The script is also resumable at the per-word level.

### 3. Evaluate
```evaluate
python evaluate.py
```
Discovers every model directory under `coded_generations/`, averages per-word frequency vectors across runs, merges against the human norms, computes per-category Pearson r and Mean r, and writes `evaluations.md` (one section per experiment, sorted by Mean r descending).

## Layout

```
supplementary/
  README.md             this file
  .env.example          template: copy to .env and add your OpenRouter key
  requirements.txt
  generate.py           free property generation for one (experiment, model)
  code.py               LLM-coding of generated properties
  evaluate.py           build evaluations.md leaderboard
  src/
    experiments.py      per-experiment config (paths, categories, stimulus loading)
    parsers.py          response/coder parsers and frequency vectors
    llm_clients/
      __init__.py       client registry (openrouter | local)
      openrouter.py     OpenAI SDK pointed at OpenRouter
      local.py          transformers + CUDA fallback
  data/
    experiment_1/
      human_norms.csv               293-row gold reference (Harpaintner et al. 2018)
      generation_prompt.txt
      coding_prompt.txt
      coding_human_ground_truth.csv expert hand-coded property labels (used to validate the LLM coder)
    experiment_2/
      README.md                     fetch instructions and license notice
      fetch_kelly.py                downloads Kelly et al. (2024) from OSF and builds the gold references
      generation_prompt.md
      coding_prompt.md
      # after running fetch_kelly.py:
      raw/                          source files from https://osf.io/eh5dk/
      human_norms.csv               235-row gold reference (Abstract + Emotion subset)
      coding_human_norms_ground_truth.csv  per-clue gold for the coding prompt
  generations/          5 sample models x 10 runs preloaded; new runs added by generate.py
  coded_generations/    5 sample models x 10 coded runs preloaded; new runs added by code.py
```
