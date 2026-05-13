"""Local CUDA client using HuggingFace transformers.

Loads the model lazily once per process and reuses it across calls. Pass
any HF model id, e.g.:
  - google/gemma-3-4b-it
  - meta-llama/Meta-Llama-3.1-8B-Instruct
  - Qwen/Qwen3-8B

Requires: torch, transformers, accelerate. A GPU is strongly recommended;
the client will fall back to CPU if CUDA is not available but inference
will be slow. Rating prompts contain ~750 words each, so context length
matters: prefer models with at least 8k context.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_cache = {}


def _load(model_name: str):
    if model_name in _cache:
        return _cache[model_name]

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32
    device_map = "auto" if has_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    _cache[model_name] = (tokenizer, model)
    return _cache[model_name]


def query_llm(prompt: str, model_name: str, seed: int = None,
              temperature: float = 1.0, max_tokens: int = 8192) -> str:
    tokenizer, model = _load(model_name)
    input_device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(input_device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
