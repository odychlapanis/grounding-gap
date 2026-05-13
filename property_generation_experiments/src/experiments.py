"""Per-experiment configuration: stimulus list, prompts, gold norms, paths.

The two experiments share a common interface so the top-level scripts
(generate.py, code.py, evaluate.py) can dispatch on the integer experiment id.

Experiment 1 (Harpaintner et al. 2018):
  - 293 abstract nouns
  - four output categories: SM, IS/E (renamed to IS_E internally), SC, VA

Experiment 2 (Kelly et al. 2024, abstract + emotion subset):
  - 236 concepts (118 Abstract + 118 Emotion)
  - four output categories: taxonomic, entity, situation, introspective
"""

import os
from dataclasses import dataclass
from typing import List

import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


@dataclass
class ExperimentConfig:
    name: str
    word_column: str
    cats: List[str]
    cat_labels: List[str]
    norms_path: str
    generation_prompt_path: str
    coding_prompt_path: str
    n_expected: int

    @property
    def generations_dir(self) -> str:
        return os.path.join(ROOT, "generations", self.name)

    @property
    def coded_dir(self) -> str:
        return os.path.join(ROOT, "coded_generations", self.name)

    def load_words(self) -> List[str]:
        df = pd.read_csv(self.norms_path)
        words = df[self.word_column].astype(str).tolist()
        if len(words) != self.n_expected:
            raise RuntimeError(
                f"{self.name}: expected {self.n_expected} stimuli, got {len(words)} in {self.norms_path}")
        return words

    def load_norms(self) -> pd.DataFrame:
        df = pd.read_csv(self.norms_path)
        if self.name == "exp1":
            df = df.rename(columns={"IS/E": "IS_E"})
        return df

    def load_generation_prompt(self) -> str:
        with open(self.generation_prompt_path) as f:
            return f.read()

    def load_coding_prompt(self) -> str:
        with open(self.coding_prompt_path) as f:
            return f.read()


_DATA = os.path.join(ROOT, "data")

_EXP1 = ExperimentConfig(
    name="exp1",
    word_column="word",
    cats=["SM", "IS_E", "SC", "VA"],
    cat_labels=["Sensorimotor", "Internal", "Social", "Verbal"],
    norms_path=os.path.join(_DATA, "experiment_1", "human_norms.csv"),
    generation_prompt_path=os.path.join(_DATA, "experiment_1", "generation_prompt.txt"),
    coding_prompt_path=os.path.join(_DATA, "experiment_1", "coding_prompt.txt"),
    n_expected=293,
)

_EXP2 = ExperimentConfig(
    name="exp2",
    word_column="concept",
    cats=["taxonomic", "entity", "situation", "introspective"],
    cat_labels=["Taxonomic", "Entity", "Situation", "Introspective"],
    norms_path=os.path.join(_DATA, "experiment_2", "human_norms.csv"),
    generation_prompt_path=os.path.join(_DATA, "experiment_2", "generation_prompt.md"),
    coding_prompt_path=os.path.join(_DATA, "experiment_2", "coding_prompt.md"),
    n_expected=235,
)


def get_config(experiment: int) -> ExperimentConfig:
    if experiment == 1:
        return _EXP1
    if experiment == 2:
        return _EXP2
    raise ValueError(f"experiment must be 1 or 2, got {experiment}")
