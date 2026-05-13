"""Generate properties (Experiment 1) or clues (Experiment 2) for one model.

Usage:
    python generate.py --experiment 1 --model openai/gpt-5.4
    python generate.py --experiment 2 --model anthropic/claude-sonnet-4-6 --runs 10
    python generate.py --experiment 1 --model google/gemma-3-4b-it --client local

Output goes to generations/exp{1,2}/{safe_model}_run_{N}.csv with columns
`word, response, properties`. Resumable: if the file exists for a given run,
already-processed words are skipped and only the missing tail is generated.
"""

import argparse
import csv
import glob
import os
import re
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import get_config
from src.llm_clients import get_client
from src.parsers import parse_response


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def existing_runs(out_dir: str, prefix: str) -> set[int]:
    runs = set()
    for f in glob.glob(os.path.join(out_dir, f"{prefix}_run_*.csv")):
        m = re.search(r"_run_(\d+)\.csv$", os.path.basename(f))
        if m:
            runs.add(int(m.group(1)))
    return runs


def already_processed(path: str) -> set[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    seen = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(row["word"])
    return seen


def run_one(experiment: int, words: list[str], prompt: str, model_name: str,
            query_llm, output_path: str, seed: int) -> None:
    seen = already_processed(output_path)
    todo = [w for w in words if w not in seen]
    if not todo:
        print(f"  run already complete: {os.path.basename(output_path)}")
        return

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["word", "response", "properties"])
        for w in tqdm(todo, desc=os.path.basename(output_path)):
            try:
                resp = query_llm(prompt.replace("{word}", str(w)), model_name=model_name, seed=seed)
            except Exception as e:
                print(f"  error on '{w}': {e}", file=sys.stderr)
                continue
            props = parse_response(experiment, resp)
            writer.writerow([w, resp, props])
            f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=int, choices=[1, 2], required=True)
    p.add_argument("--model", required=True, help="model id (OpenRouter slug or HF repo)")
    p.add_argument("--client", default="openrouter", choices=["openrouter", "local"])
    p.add_argument("--runs", type=int, default=1, help="number of new runs to launch")
    p.add_argument("--name", default=None,
                   help="output filename prefix (default: safe_name(--model)). "
                        "Use this to land new runs alongside existing files when the "
                        "filename convention differs from the resolution id.")
    args = p.parse_args()

    cfg = get_config(args.experiment)
    os.makedirs(cfg.generations_dir, exist_ok=True)

    words = cfg.load_words()
    prompt = cfg.load_generation_prompt()
    query_llm = get_client(args.client)

    prefix = args.name if args.name else safe_name(args.model)
    start = (max(existing_runs(cfg.generations_dir, prefix), default=0)) + 1

    print(f"experiment={args.experiment} model={args.model} client={args.client}")
    print(f"  stimuli: {len(words)} | output: {cfg.generations_dir}")
    print(f"  launching runs {start}..{start + args.runs - 1}")

    for i in range(start, start + args.runs):
        out_path = os.path.join(cfg.generations_dir, f"{prefix}_run_{i}.csv")
        run_one(args.experiment, words, prompt, args.model, query_llm, out_path, seed=i)


if __name__ == "__main__":
    main()
