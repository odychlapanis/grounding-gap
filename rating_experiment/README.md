# Rating experiment

Code, data, and sample runs accompanying the paper

> **The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans**

This directory reproduces the Rating Experiment. For each of the
14 ACF dimensions defined by Troche et al. (2017), the model rates 751 abstract
nouns on a 1 to 7 Likert scale. Mean per-word ratings are correlated against
the published human norms via Pearson r.

| | Rating experiment |
| :--- | :--- |
| stimuli | 751 abstract nouns (Troche et al. 2017) |
| dimensions | 14 ACF dimensions, grouped into Internal (6), Sensory (5), Magnitude (3) |
| seeds | 10 pre-shuffled word orderings shipped under `data/shuffled_dataset_seeds/` |

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

The last three lines of `requirements.txt` (`torch`, `transformers`, `accelerate`)
are commented out by default. Uncomment them only if you want to run open-weight
models on a local GPU via `--client local`; the OpenRouter path does not need them.

All API calls go through OpenRouter using the OpenAI Python SDK, so any
OpenRouter-supported model id works (e.g. `openai/gpt-5.4`,
`anthropic/claude-opus-4.6`, `google/gemini-3.1-pro`).

## Try it: see the leaderboard on the preloaded sample

The repository ships with 10 seed runs for all 21 paper models on every dimension.
Reproduce the headline leaderboard immediately, no API key required:

```evaluate
python src/evaluate.py
cat evaluations.md
```

## Reproduce or extend with a new model

The pipeline has two idempotent steps. Run them in order on any OpenRouter or
HuggingFace model id.

### 1. Generate ratings

```generate
python src/generate_ratings.py --model openai/gpt-5.4
python src/generate_ratings.py --model anthropic/claude-sonnet-4-6 --seeds 10
python src/generate_ratings.py --model google/gemma-3-4b-it --client local
python src/generate_ratings.py --model openai/gpt-5.4 --dimensions emotion social
```

For each (dimension, seed) cube, the script reads the pre-shuffled word
ordering from `data/shuffled_dataset_seeds/seed_<N>.csv`, prepends the
dimension prompt under `data/prompts/<dim>_prompt.md`, calls the model with
seed=N, and parses `('word', rating)` tuples from the response. Up to two
retries are issued for any words the model forgot to rate. Output goes to
`ratings/<safe_model>/<dimension>/seed_<N>.csv` with columns `word, rating`.

The script is resumable: re-running skips any (dimension, seed) whose output
CSV already covers every word, and within an incomplete seed it skips
already-rated words.

Pass `--name <dirname>` to override the default safe-model directory name
(useful when a new run should land alongside files written under a different
naming convention).

### 2. Evaluate

```evaluate
python src/evaluate.py
```

Discovers every model directory under `ratings/`, averages per-word ratings
across seeds for each dimension, computes Pearson r against the human norms
in `data/human_ground_truth_ratings.csv`, and writes `evaluations.md` sorted
by Mean r descending. The leaderboard reports per-dimension r as well as
Internal, Sensory, and Magnitude component means following Troche et al. (2017).

## Layout

```
rating_experiment/
  README.md              this file
  src/
    generate_ratings.py  rate every (dimension, seed) for one model
    evaluate.py          build evaluations.md leaderboard
    utils/
      parsers.py         regex parser for `('word', rating)` tuples
      llm_clients/
        __init__.py      client registry (openrouter | local)
        openrouter.py    OpenAI SDK pointed at OpenRouter
        local.py         transformers + CUDA fallback
  data/
    human_ground_truth_ratings.csv   751-row gold reference (Troche et al. 2017)
    prompts/                         14 dimension prompt templates
    shuffled_dataset_seeds/          10 pre-shuffled word orderings
  ratings/                           21 paper models x 10 seeds preloaded;
                                     new runs added by generate_ratings.py
```
