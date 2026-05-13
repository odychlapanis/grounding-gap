"""Parser for LLM rating responses.

The dimension prompts ask the model to emit one ('word', INT_SCORE) tuple per
line. We pull those tuples out with a regex and return (word, rating) pairs.
"""

import re

_TUPLE_PATTERN = re.compile(r"\(\s*'([^']+)'\s*,\s*(\d+(?:\.\d+)?)\s*\)")


def parse_rating_response(text: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for word, rating in _TUPLE_PATTERN.findall(text):
        try:
            out.append((word.strip(), float(rating)))
        except ValueError:
            continue
    return out
