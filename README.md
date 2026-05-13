# Supplementary materials

Code, data, and sample runs accompanying the paper

> **The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans**

Directories:
| Directory | What it reproduces |
| :--- | :--- |
| [`property_generation_experiments/`](property_generation_experiments/) | Experiments 1 and 2: free property generation (Harpaintner et al. 2018; Kelly et al. 2024). |
| [`rating_experiment/`](rating_experiment/) | Rating Experiment: 14-dimension Likert rating (Troche et al. 2017).|
| [`mechanistic_analysis/`](mechanistic_analysis/) | SAE feature analysis and steering artifacts. |

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

The Kelly et al. (2024) data used in Experiment 2 is not bundled (see [`DATA_LICENSES.md`](DATA_LICENSES.md)). One additional step fetches it from the authors' OSF project:

```fetch
python property_generation_experiments/data/experiment_2/fetch_kelly.py
```

## Try it: reproduce the leaderboards on the preloaded sample

No API key required for the evaluation step.

```evaluate
# Experiments 1 and 2 (property generation)
cd property_generation_experiments
python evaluate.py
cat evaluations.md

# Rating Experiment
cd ../rating_experiment
python src/evaluate.py
cat evaluations.md
```

Each leaderboard reproduces the corresponding paper table.

## Layout

```
supplementary/
  README.md                          this file
  LICENSE                            Apache-2.0 covering this repository's code
  DATA_LICENSES.md                   per-dataset notice for the third-party norms
  requirements.txt                   shared Python deps for all experiments
  .env.example                       template: copy to .env and add your OpenRouter key
  property_generation_experiments/   Experiments 1 and 2
  rating_experiment/                 Rating Experiment
  mechanistic_analysis/              SAE feature analysis
```

Each subdirectory has its own README with a per-experiment walkthrough.

## Citation

```bibtex
@misc{chlapanis2026groundinggapllmsanchor,
      title={The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans},
      author={Odysseas S. Chlapanis and Orfeas Menis Mastromichalakis and Christos H. Papadimitriou},
      year={2026},
      eprint={2605.08837},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.08837},
}
```

## License

Source code in this repository is released under Apache License 2.0 (see [`LICENSE`](LICENSE)). Third-party datasets retain their original licenses (Harpaintner 2018 and Troche 2017 under CC-BY 4.0; Kelly 2024 fetched on demand from OSF and not redistributed). See [`DATA_LICENSES.md`](DATA_LICENSES.md) for the full per-dataset notice.
