"""Run a model over every (dimension, seed) cube of the Troche rating benchmark.

For each of the 14 dimensions in `data/prompts/`, this script:

1. Loads a pre-shuffled word ordering from `data/shuffled_dataset_seeds/seed_<N>.csv`.
2. Appends the shuffled list to the dimension's prompt template.
3. Calls the model through OpenRouter (or a local CUDA backend) with seed=N.
4. Parses ('word', rating) tuples from the reply.
5. Retries up to 2 times for any words the model forgot to rate.
6. Writes per-word ratings to
   `ratings/<safe_model>/<dimension>/seed_<N>.csv` with columns `word, rating`.

The script is resumable: re-running skips any (dimension, seed) whose output
CSV already covers every word.

Usage (run from the rating_experiment/ directory):
    python src/generate_ratings.py --model openai/gpt-5.4
    python src/generate_ratings.py --model google/gemma-3-4b-it --client local --seeds 5
    python src/generate_ratings.py --model anthropic/claude-sonnet-4-6 --dimensions emotion social
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.llm_clients import get_client
from utils.parsers import parse_rating_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(ROOT, "data", "prompts")
SEED_DIR = os.path.join(ROOT, "data", "shuffled_dataset_seeds")
RATINGS_ROOT = os.path.join(ROOT, "ratings")
MAX_RETRIES = 2


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def load_dimensions() -> list[str]:
    return sorted(
        f[:-len("_prompt.md")]
        for f in os.listdir(PROMPT_DIR)
        if f.endswith("_prompt.md")
    )


def load_seed(n: int) -> list[str]:
    df = pd.read_csv(os.path.join(SEED_DIR, f"seed_{n}.csv"))
    return df["word"].astype(str).str.strip().tolist()


def read_prompt(dimension: str) -> str:
    with open(os.path.join(PROMPT_DIR, f"{dimension}_prompt.md")) as f:
        return f.read()


def already_rated(path: str) -> dict[str, float]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    df = pd.read_csv(path)
    df = df.dropna(subset=["rating"])
    return dict(zip(df["word"].astype(str).str.strip(), df["rating"].astype(float)))


def run_one_seed(model_name: str, query_llm, dimension: str, prompt_template: str,
                 words: list[str], seed: int, out_path: str) -> None:
    collected = already_rated(out_path)
    target_set = set(words)
    missing = [w for w in words if w not in collected]
    if not missing:
        return

    for attempt in range(MAX_RETRIES + 1):
        if not missing:
            break
        prompt = f"{prompt_template}\nWords:\n" + "\n".join(missing)
        try:
            raw = query_llm(prompt, model_name=model_name, seed=seed)
        except Exception as e:
            print(f"  error on {dimension} seed {seed}: {e}", file=sys.stderr)
            break
        for word, rating in parse_rating_response(raw):
            if word in target_set and word not in collected:
                collected[word] = rating
        missing = [w for w in words if w not in collected]
        if missing and attempt < MAX_RETRIES:
            time.sleep(1.5)

    out_df = pd.DataFrame(
        [(w, collected.get(w, np.nan)) for w in words],
        columns=["word", "rating"],
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model id (OpenRouter slug or HF repo)")
    p.add_argument("--client", default="openrouter", choices=["openrouter", "local"])
    p.add_argument("--seeds", type=int, default=10, help="number of seeds (1..N) to run")
    p.add_argument("--dimensions", nargs="*", default=None,
                   help="subset of dimensions to run (default: all 14)")
    p.add_argument("--name", default=None,
                   help="output dirname under ratings/ (default: safe_name(--model))")
    args = p.parse_args()

    dimensions = args.dimensions or load_dimensions()
    query_llm = get_client(args.client)

    out_dir = os.path.join(RATINGS_ROOT, args.name if args.name else safe_name(args.model))

    print(f"model={args.model} client={args.client}")
    print(f"  dimensions: {len(dimensions)} | seeds: {args.seeds}")
    print(f"  output: {out_dir}")

    for dimension in dimensions:
        prompt_template = read_prompt(dimension)
        for seed in tqdm(range(1, args.seeds + 1), desc=dimension):
            words = load_seed(seed)
            out_path = os.path.join(out_dir, dimension, f"seed_{seed}.csv")
            run_one_seed(args.model, query_llm, dimension, prompt_template,
                         words, seed, out_path)


if __name__ == "__main__":
    main()
