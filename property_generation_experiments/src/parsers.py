"""Output parsers for the property-generation and coding prompts.

Experiment 1 (Harpaintner)
  - generate prompt asks for: "word: X / properties: a, b, c, d"
  - coding prompt asks for one line per property: "<property>: <Category>"

Experiment 2 (Kelly)
  - generate prompt asks for 5 numbered clues: "1. clue_one\n2. clue_two..."
  - coding prompt asks for a JSON array of objects with subordinate +
    superordinate labels.
"""

import json
import re
from typing import List, Tuple

EXP1_LABEL_MAP = {
    "sensorimotor feature": "SM",
    "internal state and emotion": "IS_E",
    "social constellation": "SC",
    "association": "VA",
    "other abstract concept": "VA",
}
EXP1_CATS = ["SM", "IS_E", "SC", "VA"]

EXP2_CATS = ["taxonomic", "entity", "situation", "introspective"]


# ---------------------------------------------------------------------------
# Generation parsers
# ---------------------------------------------------------------------------

def parse_exp1_response(text: str) -> List[str]:
    """Return the four properties from a Harpaintner-style response."""
    m = re.search(r"properties\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if not m:
        return []
    line = m.group(1).splitlines()[0]
    return [p.strip() for p in line.split(",") if p.strip()]


def parse_exp2_response(text: str) -> List[str]:
    """Return the five clues from a Kelly-style numbered-list response."""
    return [m.strip() for m in re.findall(r"^\s*\d+[\.\)]\s*(.+)", text, flags=re.MULTILINE)]


def parse_response(experiment: int, text: str) -> List[str]:
    return parse_exp1_response(text) if experiment == 1 else parse_exp2_response(text)


# ---------------------------------------------------------------------------
# Coding parsers
# ---------------------------------------------------------------------------

def parse_exp1_codes(text: str) -> List[str]:
    """Map each `<property>: <Category>` line to one of EXP1_CATS."""
    out = []
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        label = line.split(":", 1)[1].strip().lower()
        for key, code in EXP1_LABEL_MAP.items():
            if key in label:
                out.append(code)
                break
    return out


def parse_exp2_codes(text: str) -> List[str]:
    """Extract the lower-cased `superordinate` field from a Kelly JSON array."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    out = []
    for entry in data:
        sup = str(entry.get("superordinate", "")).strip().lower()
        if sup in EXP2_CATS:
            out.append(sup)
    return out


def parse_codes(experiment: int, text: str) -> List[str]:
    return parse_exp1_codes(text) if experiment == 1 else parse_exp2_codes(text)


# ---------------------------------------------------------------------------
# Frequency vector
# ---------------------------------------------------------------------------

def codes_to_frequencies(experiment: int, codes: List[str]) -> Tuple[List[float], List[str]]:
    """Return (per-category proportion vector, ordered category names)."""
    cats = EXP1_CATS if experiment == 1 else EXP2_CATS
    n = len(codes)
    if n == 0:
        return [0.0] * len(cats), cats
    return [codes.count(c) / n for c in cats], cats
