# Supplementary materials

Code, data, and sample runs accompanying the paper

> **The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans**

Directories:
| Directory | What it reproduces |
| :--- | :--- |
| [`property_generation_experiments/`](property_generation_experiments/) | Experiments 1 and 2: free property generation (Harpaintner et al. 2018; Kelly et al. 2024). 21 paper models x 10 runs preloaded for Exp 1; 5 sample models x 10 runs for Exp 2. |
| [`rating_experiment/`](rating_experiment/) | Experiment 3: 14-dimension Likert rating (Troche et al. 2017). 21 paper models x 10 seeds preloaded. |
| [`mechanistic_analysis/`](mechanistic_analysis/) | SAE feature analysis and steering artifacts referenced in Section 5 of the paper. |

## Installation

The shipped `requirements.txt` and `.env.example` at the supplementary root cover every script under every experiment.

```setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
$EDITOR .env  # paste your OPENROUTER_API_KEY
```

The last three lines of `requirements.txt` (`torch`, `transformers`, `accelerate`) are commented out by default. Uncomment them only if you want to run open-weight models on a local GPU via `--client local`; the OpenRouter path does not need them.

All API calls go through OpenRouter using the OpenAI Python SDK, so any OpenRouter-supported model id works (e.g. `openai/gpt-5.4`, `anthropic/claude-opus-4.6`, `google/gemini-3.1-pro`).

## Try it: reproduce the leaderboards on the preloaded sample

No API key required for the evaluation step.

```evaluate
# Experiments 1 and 2 (property generation)
cd property_generation_experiments
python evaluate.py
cat evaluations.md

# Experiment 3 (rating)
cd ../rating_experiment
python src/evaluate.py
cat evaluations.md
```

Each leaderboard reproduces the corresponding paper table (Table A1 for property generation, Table A2 for ratings) within bootstrap noise.

## Layout

```
supplementary/
  README.md                          this file
  requirements.txt                   shared Python deps for all experiments
  .env.example                       template: copy to .env and add your OpenRouter key
  property_generation_experiments/   Experiments 1 and 2
  rating_experiment/                 Experiment 3
  mechanistic_analysis/              SAE feature analysis
```

Each subdirectory has its own README with a per-experiment walkthrough.
