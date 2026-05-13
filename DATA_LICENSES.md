# Data licenses

This repository's own source code is released under Apache-2.0 (see
`LICENSE`). The third-party datasets used by the experiments retain their
original licenses, listed below.

## Experiment 1: Harpaintner et al. (2018)

- **Dataset**: property listings and category coding for 296 abstract words.
- **Citation**:
  > Harpaintner, M., Trumpp, N. M., & Kiefer, M. (2018). The Semantic
  > Content of Abstract Concepts: A Property Listing Study of 296 Abstract
  > Words. *Frontiers in Psychology*, 9, 1748.
- **Source**: <https://doi.org/10.3389/fpsyg.2018.01748>
- **License**: **Creative Commons Attribution 4.0 International (CC-BY 4.0)**
  (Frontiers is a fully open-access publisher; all article content and
  supplementary materials are released under CC-BY 4.0.)
- **Bundled in**: `property_generation_experiments/data/experiment_1/`
  (`human_norms.csv`, `coding_human_ground_truth.csv`). Redistributed under
  CC-BY 4.0 with attribution above.

## Experiment 2: Kelly et al. (2024)

- **Dataset**: feature norms for 235 Abstract + Emotion concepts (plus 119
  concrete controls in the source).
- **Citation**:
  > Kelly, A. E., et al. (2024). Conceptual Structure of Emotions. *Emotion*.
- **Source**: <https://osf.io/eh5dk/>
- **License**: the OSF project is **public but carries no explicit license**
  (`node_license: null`). The authors retain copyright.
- **Bundled in**: nothing. We do **not** redistribute Kelly et al.'s data.
  The file `property_generation_experiments/data/experiment_2/fetch_kelly.py`
  downloads the source files from the authors' canonical OSF page on each
  user's machine and builds the per-clue and per-concept gold references
  used by `evaluate.py`. Users should cite Kelly et al. (2024) when using
  this experiment.

## Experiment 3: Troche et al. (2017)

- **Dataset**: 14-dimension Likert ratings for 750 words spanning
  concreteness.
- **Citation**:
  > Troche, J., Crutch, S. J., & Reilly, J. (2017). Defining a Conceptual
  > Topography of Word Concreteness: Clustering Properties of Emotion,
  > Sensation, and Magnitude among 750 English Words. *Frontiers in
  > Psychology*, 8, 1787.
- **Source**: <https://doi.org/10.3389/fpsyg.2017.01787>
- **License**: **Creative Commons Attribution 4.0 International (CC-BY 4.0)**
  (Frontiers open access; see Experiment 1 note above).
- **Bundled in**: `rating_experiment/` (human ground-truth ratings + shuffled
  seed lists). Redistributed under CC-BY 4.0 with attribution above.

## Mechanistic analysis

- **Gemma-3 model weights** (`google/gemma-3-{1b,4b,12b}-it`): governed by
  the **Gemma Terms of Use** (<https://ai.google.dev/gemma/terms>). Gated
  HuggingFace access; requires an HF token plus terms acceptance. We do not
  redistribute Gemma weights.
- **GemmaScope SAEs** (`google/gemma-scope-*`): Apache-2.0.
- **SAELens** (<https://github.com/jbloomAus/SAELens>): MIT.
- **TransformerLens** (<https://github.com/TransformerLensOrg/TransformerLens>):
  MIT.
- The shipped activation caches under `mechanistic_analysis/activations/`
  are author-generated derivatives of running these models on our stimulus
  pool and are released under Apache-2.0 with this repository.
