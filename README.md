# Decode & Detect submission: XRPLarp (version 12, blend of versions 9 and 11)

**Team:** XRPLarp. Captain: Louis Razumovsky-Hough. Members: Shaan Fisher, Arthur Clark. Every member has accepted the hackathon terms individually.
**Version submitted:** version 12 (this ZIP, `final final submission`), predictions `predictions_v12.csv`. Code released under the MIT licence.
**Seeds:** both pipelines use `SEEDS = (0, 1, 2, 3, 4)` in `v9/predict.py` and `v11/predict.py`; link model `random_state=0`; the blend has no randomness.
**Dependencies:** `requirements.txt` (Python 3.12; pandas, numpy, scipy, pyarrow, lightgbm, xgboost, scikit-learn; catboost listed but unused by default).
**External services / compute:** none at prediction time; CPU only, about 40 minutes.
**Also in this ZIP:** `XRPLarp_method.pdf` (method, 4 pages), `RESULTS.md`, `PROMPTS.md`.

Scores every test transaction with a suspicion score from 0 to 1.

## Exact command

Unzip the training package so that `train/` sits at `train_pkg/train`, and the test package into `test_pkg/`. Then:

```sh
pip install -r requirements.txt
python predict.py --train train_pkg/train --test test_pkg --out predictions.csv
```

This runs `v9/predict.py` and `v11/predict.py` unchanged, each in its own process, writing
`blend_work/predictions_v9.csv` and `blend_work/predictions_v11.csv`, then blends them with `blend.py`.
Runtime: about 40 minutes on a laptop CPU (two ~20-minute pipelines), under 4 GB RAM, no GPU. Python 3.12.
`--parallel` runs both pipelines at once (about 8 GB RAM). Results are deterministic (fixed seeds in both pipelines; the blend has no randomness).

If the two version CSVs already exist, the blend alone takes seconds:

```sh
python blend.py --a predictions_v9.csv --b predictions_v11.csv --out predictions.csv
```

## Method

1. **Version 9** (`v9/`, leaderboard 73.4): v8 features + soft cumulative balances, account balance trajectories and typed neighbour channels. See `v9/README.md`.
2. **Version 11** (`v11/`, leaderboard 73.3): v8 features + second-hop neighbour features. See `v11/README.md`.
3. **Blend** (`blend.py`): each version's scores are converted to normalised ranks (average rank for ties, divided by the number of rows) and averaged with equal weights. The metric (average precision) depends only on the ordering; the two versions' scores are on different scales in the low tail, so ranks combine the two orderings without one version's scale dominating. Both versions share the same base pipeline but differ in their extra feature families (v9: balance and typed-neighbour features; v11: second-hop neighbour features), so their errors are partly independent. `--method mean` and `--method logit` are the alternative blends recorded in `VARIANT.txt`.

## Use of the training labels, what we did not write ourselves

Unchanged from versions 9 and 11; see `v9/README.md` and `v11/README.md`. The blend step uses no labels.

## AI disclosure (model, provider, prompts, settings)

- **Model and provider:** Claude Fable and Claude Opus 5.5 (Anthropic), used through the claude.ai app; GPT Astra (OpenAI), used through chatgpt.com.
- **Settings:** default app settings. No fine-tuning, no custom system prompts, no API use.
- **Prompts:** see `PROMPTS.md` (share links to every conversation).
- **Reproducibility limits:** re-sending the prompts would not give identical code, because LLM output varies. The code here is the fixed record and reproduces the predictions without any AI service.
- The code was written mainly by Claude, with some help from GPT Astra, at our team's direction during the event. The team chose the approach, reviewed the results and is responsible for the submission.

## Limitations

- Synthetic data only: a high score here does not show the method works on real financial crime, and this is not a certified compliance tool.
- Hard multi-round relay cases are largely missed; about 1 in 5 top-ranked payments is a false positive, so results need human review.
- Signals such as first-time pairs, regular timing and night-time activity reflect how this data was generated and could unfairly flag new or automated users on real data.
- The v12 blend weights (equal) were not tuned or validated locally.

## Licence

MIT (see `LICENSE`).
