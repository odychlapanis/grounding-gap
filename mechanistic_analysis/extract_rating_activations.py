"""Extract last-token SAE feature activations on the Troche 751-word pool.

For each prompt variant in `data/prompts/prompt{N}.md`, runs a Gemma model and
encodes the residual stream at every configured layer through the matching
GemmaScope SAE. Stores only non-zero feature activations.

Output (one JSON per prompt):
    activations/<variant>_<model_tag>/prompt<N>.json
Schema:
    {word: {layer_str: {feat_id_str: rounded_float}}}

The script is resumable per (word, layer): re-running tops up missing entries
without re-extracting cached ones. Runs all 3 prompts by default; pass
`--prompts 1` (etc) to restrict.

Usage (from mechanistic_analysis/, with libs/SAELens and libs/TransformerLens
on PYTHONPATH; see README.md):
    python extract_rating_activations.py --config configs/4b_resid_post.json
    python extract_rating_activations.py --config configs/12b_resid_post.json --prompts 1 2
    python extract_rating_activations.py --config configs/4b_resid_post_all.json
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def normalise(w: str) -> str:
    return str(w).strip().lower()


def load_words(troche_csv: str) -> list[str]:
    import pandas as pd
    df = pd.read_csv(troche_csv)
    return sorted({normalise(w) for w in df["word"].dropna()})


def build_prompt(word: str, prompt_path: str) -> str:
    """Compose the chat-formatted Gemma input. The model prefix ends at the
    position whose activation we want to encode (last token = colon after the
    output-format keyword). prompt1/prompt2 ask for `properties:`, prompt3 for
    `words:` -- the prefix mirrors that.
    """
    body = open(prompt_path).read().rstrip("\n").replace("{word}", word)
    keyword = "Words" if "words: <word1>" in body.lower() else "Properties"
    return (
        f"<start_of_turn>user\n{body}<end_of_turn>\n"
        f"<start_of_turn>model\nWord: {word}\n{keyword}:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config JSON")
    ap.add_argument("--prompts", nargs="*", type=int, default=[1, 2, 3],
                    choices=[1, 2, 3], help="prompt ids to run (default: all 3)")
    ap.add_argument("--troche_csv", default=None,
                    help="override path to the 751-word ratings CSV "
                         "(default: ../rating_experiment/data/human_ground_truth_ratings.csv)")
    ap.add_argument("--max_words", type=int, default=None, help="cap for smoke testing")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer subset overriding the config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    variant = cfg["variant"]
    model_tag = cfg["model_tag"]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else cfg["layers_to_test"]
    sae_id_tmpl = cfg.get("sae_id_template",
                          f"layer_{{}}_width_{cfg['sae_width']}_l0_{cfg['sae_l0']}")

    troche_csv = args.troche_csv or os.path.join(
        HERE, "..", "rating_experiment", "data", "human_ground_truth_ratings.csv")
    troche_csv = os.path.abspath(troche_csv)

    words = load_words(troche_csv)
    if args.max_words:
        words = words[: args.max_words]

    print(f"config:  {args.config}")
    print(f"model:   {cfg['model_name']}  (tag={model_tag})")
    print(f"variant: {variant}")
    print(f"layers:  {layers}")
    print(f"prompts: {args.prompts}")
    print(f"words:   {len(words)} from {troche_csv}")

    import torch
    from tqdm import tqdm
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    n_devices = cfg.get("n_devices", 1)
    actual = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_nd = min(n_devices, actual) if actual > 0 else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {cfg['model_name']} (n_devices={use_nd}/{actual}) ...")

    model = HookedTransformer.from_pretrained(
        cfg["model_name"],
        n_devices=use_nd,
        dtype=torch.bfloat16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    torch.set_grad_enabled(False)
    print("Model loaded.")

    out_dir = os.path.join(HERE, "activations", f"{variant}_{model_tag}")
    os.makedirs(out_dir, exist_ok=True)

    caches: dict[int, dict] = {}
    for p in args.prompts:
        out_path = os.path.join(out_dir, f"prompt{p}.json")
        if os.path.exists(out_path):
            cached = json.load(open(out_path))
            caches[p] = {normalise(k): v for k, v in cached.items()}
            print(f"  prompt{p}: resuming with {len(caches[p])} cached words")
        else:
            caches[p] = {}
        for w in words:
            caches[p].setdefault(w, {})

    for layer in layers:
        hook_name = cfg["hook_point_template"].format(layer)
        sae_id = sae_id_tmpl.format(layer)
        print(f"\n--- Layer {layer} | SAE: {sae_id} ---")

        attempt = 0
        while True:
            try:
                result = SAE.from_pretrained(release=cfg["sae_release"],
                                             sae_id=sae_id, device=device)
                sae = (result[0] if isinstance(result, tuple) else result).to(torch.bfloat16)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    raise
                backoff = min(2 ** attempt * 10, 300)
                print(f"  SAE load failed ({type(e).__name__}: {e}); retry in {backoff}s")
                time.sleep(backoff)

        layer_str = str(layer)
        for p in args.prompts:
            prompt_path = os.path.join(HERE, "data", "prompts", f"prompt{p}.md")
            cache = caches[p]
            todo = [w for w in words if layer_str not in cache[w]]
            if not todo:
                continue
            for word in tqdm(todo, desc=f"prompt{p} L{layer}", leave=False):
                full_prompt = build_prompt(word, prompt_path)
                tokens = model.to_tokens(full_prompt)
                _, act_cache = model.run_with_cache(tokens, names_filter=[hook_name])
                resid = act_cache[hook_name][0, -1, :].to(device).to(torch.bfloat16)
                feats = sae.encode(resid.unsqueeze(0)).squeeze(0).float()
                nz = (feats > 0).nonzero(as_tuple=True)[0].tolist()
                cache[word][layer_str] = {
                    str(i): round(float(feats[i].item()), 4) for i in nz
                }
            out_path = os.path.join(out_dir, f"prompt{p}.json")
            with open(out_path, "w") as f:
                json.dump(cache, f)
            print(f"  flushed {out_path}")

        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Activations written under {out_dir}/prompt{{1,2,3}}.json")


if __name__ == "__main__":
    main()
