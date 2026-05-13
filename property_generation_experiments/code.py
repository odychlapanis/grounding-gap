"""Code (label) the generated properties with an LLM coder.

Usage:
    python code.py --experiment 1 --model openai/gpt-5.4
    python code.py --experiment 1 --model openai/gpt-5.4 --coder google/gemini-2.5-flash-lite
    python code.py --experiment 2 --model anthropic/claude-sonnet-4-6 --coder google/gemini-2.5-flash

Defaults follow the paper: gemini-2.5-flash-lite codes Experiment 1
generations, gemini-2.5-flash codes Experiment 2 generations. Both are
called via OpenRouter. Output goes to
coded_generations/exp{1,2}/{safe_model}/{safe_model}_coded_with_{safe_coder}_run_{N}.csv
with columns `word, properties, codes, frequencies`.
"""

import argparse
import ast
import csv
import glob
import os
import re
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import get_config
from src.llm_clients import get_client
from src.parsers import codes_to_frequencies, parse_codes

DEFAULT_CODERS = {
    1: "google/gemini-2.5-flash-lite",
    2: "google/gemini-2.5-flash",
}


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def discover_runs(generations_dir: str, model_prefix: str) -> dict[int, str]:
    files = glob.glob(os.path.join(generations_dir, f"{model_prefix}_run_*.csv"))
    out = {}
    for f in files:
        m = re.search(r"_run_(\d+)\.csv$", os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def already_coded_words(path: str) -> set[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    seen = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seen.add(row["word"])
    return seen


def normalize_properties(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(p).strip() for p in raw if str(p).strip()]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except (ValueError, SyntaxError):
            pass
        return [p.strip() for p in raw.strip("[]").split(",") if p.strip()]
    return []


def code_one_run(experiment: int, gen_path: str, out_path: str, prompt: str,
                 coder_model: str, query_llm, seed: int) -> None:
    seen = already_coded_words(out_path)
    rows_to_write = []

    with open(gen_path, newline="") as f:
        gen_rows = list(csv.DictReader(f))

    todo = [r for r in gen_rows if r["word"] not in seen]
    if not todo:
        print(f"  run already coded: {os.path.basename(out_path)}")
        return

    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["word", "properties", "codes", "frequencies"])
        for r in tqdm(todo, desc=os.path.basename(out_path)):
            word = r["word"]
            props = normalize_properties(r.get("properties", ""))
            if not props:
                writer.writerow([word, props, [], [0.0] * 4])
                continue
            user_prompt = prompt.replace("{word}", str(word)).replace("{properties}", ", ".join(props))
            try:
                resp = query_llm(user_prompt, model_name=coder_model, seed=seed, temperature=0.0)
            except Exception as e:
                print(f"  error on '{word}': {e}", file=sys.stderr)
                continue
            codes = parse_codes(experiment, resp)
            freqs, _ = codes_to_frequencies(experiment, codes)
            writer.writerow([word, props, codes, freqs])
            f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=int, choices=[1, 2], required=True)
    p.add_argument("--model", required=True, help="model whose generations to code")
    p.add_argument("--coder", default=None, help="coder model id (default per experiment)")
    p.add_argument("--client", default="openrouter", choices=["openrouter", "local"])
    p.add_argument("--runs", type=int, default=None,
                   help="cap on number of runs to code (default: all available)")
    p.add_argument("--name", default=None,
                   help="filename prefix that identifies the model whose generations to code "
                        "(default: safe_name(--model)).")
    p.add_argument("--coder-name", default=None,
                   help="filename prefix for the coder portion of the output filename "
                        "(default: safe_name(--coder)).")
    args = p.parse_args()

    cfg = get_config(args.experiment)
    coder = args.coder or DEFAULT_CODERS[args.experiment]
    query_llm = get_client(args.client)
    prompt = cfg.load_coding_prompt()

    model_prefix = args.name if args.name else safe_name(args.model)
    coder_prefix = args.coder_name if args.coder_name else safe_name(coder)

    out_dir = os.path.join(cfg.coded_dir, model_prefix)
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_runs(cfg.generations_dir, model_prefix)
    if not runs:
        sys.exit(f"no generations found at {cfg.generations_dir} for prefix {model_prefix}")

    selected = list(runs.items())
    if args.runs is not None:
        selected = selected[:args.runs]

    print(f"experiment={args.experiment} model={args.model} coder={coder} client={args.client}")
    print(f"  coding {len(selected)}/{len(runs)} run(s) -> {out_dir}")

    for run_idx, gen_path in selected:
        out_path = os.path.join(out_dir, f"{model_prefix}_coded_with_{coder_prefix}_run_{run_idx}.csv")
        code_one_run(args.experiment, gen_path, out_path, prompt, coder, query_llm, seed=run_idx)


if __name__ == "__main__":
    main()
