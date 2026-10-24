# Decode & Detect submission: XRPLarp (version 11)

Scores every test transaction with a suspicion score from 0 to 1.

## Exact command

Unzip the training package so that `train/` sits at `train_pkg/train`, and the test package into `test_pkg/`. Then:

```sh
pip install -r requirements.txt
python predict.py --train train_pkg/train --test test_pkg --out predictions.csv
```

Runtime: about 20 minutes on a laptop CPU, under 4 GB RAM, no GPU. Python 3.12.

The constants `BLEND`, `SMOOTH_MODE`, `SMOOTH_WEIGHT` and `PWEIGHT_MIN` at the top of `predict.py` select the post-processing variant. The defaults (`linear`, `pweighted`, 0.85, 0.5) produce the uploaded CSV; `VARIANT.txt` records which CSV was made with which settings.
Results are deterministic (fixed seeds); we re-ran the command from a clean folder and got identical scores.

Local validation (repeated grouped 5-fold CV on `train/`, same metric as the site):

```sh
python validate.py --train train_pkg/train
```

## Method

1. **`load.py`**: one tidy row per payment: sender, destination, delivered asset and amount, the first executed leg (actual source debit), and USD values using the hourly reference prices for each asset (issuer + currency).
2. **`features.py`**: about 195 features per payment, all computed within one split (wallets do not overlap between splits):
   - **Relationship novelty:** how often this sender→receiver pair occurs in the week. Planted schemes almost always use first-time pairs.
   - **Bursts:** how many payments the sender and receiver make within 2 minutes to 8 hours around this payment, for all payments and for first-time pairs only.
   - **Temporal motif components:** payments are linked when they share an account and happen within a gap (3 minutes to 3 hours). Statistics of the resulting group (size, accounts, span, spacing, maximum fan-in/out, loops, currencies, amount spread and trend) describe the shape the payment belongs to.
   - **Money-moved-on links:** a first-time-pair payment followed within 1–4 hours by a payment from its receiver, with similar amounts (chains) or pieces that add up (splits and merges).
   - Amount, conversion and timing-to-next-payment features.
3. **`predict.py`**: two models trained on all of `train/`, each an average of LightGBM (7 leaves) and XGBoost (depth 3) over 5 seeds (CatBoost is available in the code but off):
   - **(a)** suspicious vs not;
   - **(b)** "part of any scheme-shaped group", suspicious or innocent look-alike, which learns what slow schemes look like from twice as many examples.

   Each model's score is blended with the mean score of the payment's linked groups (first-time-pair bursts at 10 and 20 minute gaps, and either rebuilt groups at 0.9 link confidence or, in version 6's `pweighted` mode, a link-probability-weighted mean over the payment's links), since all payments in a scheme share a label. Final score = (a) + 0.25 × (b) (`linear`), or (a) + 0.25 × (b) × (1 − (a)) (`rankaware`, version 6: the shape model lifts uncertain payments without reordering the confident ones), rescaled to 0–1.
4. **Rebuilt groups** (`links.py`, `groupfeat.py`, added in version 4): a LightGBM link model learns, from the training split, whether two payments that share an account within 8 hours belong to the same planted scenario (inputs: time gap, direction, pair rarity, amount ratio, currencies, legs, account activity). On held-out training scenarios it reaches 0.99 AUC. Confident links (p > 0.7, 0.8, 0.9) are clustered into groups, and each payment gets its group's properties: size, span, spacing, amounts, currencies, share of first-time pairs, number of middle accounts, whether middle accounts ever pay out more than they have received so far, and how much of what they receive passes through. For the training rows, link probabilities are out-of-fold (two halves split by scenario), so training features look like test features.
5. **Cross-round links and per-account groups** (`links.py`, `egofeat.py`, added in version 5): relay schemes run in rounds 1–6 days apart, well beyond the 8-hour window, and repeat the same sender→receiver payment across rounds. The link model therefore also scores each payment against the next one or two payments of the same pair. Then, for every account, only that account's own payments are clustered, joined when a confident link (p > 0.8, 0.9) runs through that account. This rebuilds multi-round relay groups without chaining across accounts into giant clusters. Each payment gets its sender's and its receiver's group: size, rounds, span, inflows and outflows, whether the account ever pays out more than it has received so far (funded), and how much passes through. Version 4's group features and smoothing are computed as before from the within-8-hour links only, but their link probabilities come from the retrained link model, so they shift slightly.
6. **Soft funding features** (`softflow.py`, version 6.1): rebuilt groups break the relay "funded" test, because one wrong member makes a funded group look unfunded. Instead each within-8-hour link probability is used as a soft "same scenario" weight between two payments at one account. For a payment out of an account: the probable same-scenario money that arrived before it (funded) and only after it (paid out before being funded, the look-alike's tell), the amount-weighted versions and their difference; for a payment into an account: the mirror (passed on afterwards vs already paid out before). Ten features, no clustering.
7. **Version 8: neighbour and spread features** (`neighfeat.py`): every payment receives the link-probability-weighted mean of key features of the payments it is linked to (soft funding, novelty, burst counts, time-to-forward), computed separately over within-8-hour links, over the subset of those where money came into the shared account before leaving it, and over cross-round same-pair links. The last carries soft-funding evidence across relay rounds days apart, which no within-8-hour feature can express (one-hop feature aggregation as in Weber et al. 2019; neighbourhood features as in Savage et al. 2016). Gap-regularity features (standard deviation, coefficient of variation, minimum and maximum of inter-payment gaps) are added per temporal component and per per-account group, since planted schedules are more regular than organic traffic.
8. **Version 6** changes no model and adds two features and two post-processing options: the number of distinct senders and receivers per temporal motif component (computed since version 1 but not previously output), a night-time flag (00:00–06:00; planted motifs are spread evenly through the day while background traffic is daytime-heavy), the `rankaware` blend and the `pweighted` smoothing described above.

Local CV (100 × AP, grouped 5-fold CV): version 4 61.7, version 5 61.1–61.5, version 6 62.3 (two repeats, ±0.1); version 7 was uploaded on the strength of versions 6 and 6.1 scoring independently without a separate local run. Local scores move about ±1 point between feature changes. Leaderboard history: version 1 (model (a), LightGBM only) 65.5; version 2 (+ model (b)) 67.1; version 3 (+ XGBoost in the average) 67.4; version 4 (+ rebuilt groups) 70.3; version 5 (+ cross-round links and per-account groups) 71.2; version 6.1 (+ soft funding features, point 6) 71.9; version 6 (+ point 7) 72.0; version 7 (points 6 and 8 combined) 72.4; version 8 (point 7) 73.1; version 10 (CatBoost added to the average) 72.9, neutral, reverted; version 11 (this code) adds a second hop of neighbour aggregation: the version 8 neighbour features are themselves averaged over a payment's links with probability weights. Easy cases are mostly caught; medium cases partly; the hard relay cases essentially not.

## Use of the training labels

- `is_suspicious` is the training target of model (a).
- `scenario_id` is used for validation (`validate.py` keeps every scenario inside one cross-validation fold) and **to train the link model** in `links.py` (target: do two training payments belong to the same scenario?), whose links feed both the version 4 groups and the version 5 per-account groups. It is never an input to any model, and no label is used when scoring the test data.
- `family` is used **only to define the training target of model (b)**: 1 if a training payment belongs to a planted scheme-shaped scenario (suspicious or innocent look-alike), 0 otherwise. It is never a model input.
- No label column is available or used when scoring the test data; test scores come only from features computed on the test transactions.

## What we did not write ourselves

- **AI assistance:** the code in this repository was written by Claude (Anthropic's AI assistant) at our team's direction during the event, including the data analysis that shaped the features. Our team chose the approach, reviewed the analysis, ran and uploaded the results, and is responsible for the submission.
- **Pre-existing work:** on 23 October, before the event, we prepared a practice pipeline with Claude on synthetic data of our own. None of its files are included here, but some ideas carried over: the sort-and-search windowed counting helper, burst/"follow the money" features, and blending scores across linked payments.
- **Organisers' material:** `load.py` follows the data conventions in the organisers' README and `payments.py` (USD valuation via `market_rates.csv`, first executed leg as actual source debit). Some feature ideas follow the organisers' `starter_notebook.ipynb` (for example: the receiver forwards money within an hour; bursts of payments by one account). No organiser code is copied.
- **Libraries:** pandas, numpy, scipy, pyarrow, LightGBM, XGBoost, CatBoost, scikit-learn (versions in `requirements.txt`; their licences apply).
- **Published ideas:** group-level pass-through and "middle accounts retain almost nothing" follow FlowScope (Li et al., AAAI 2020); scoring whole rebuilt groups follows Savage et al. (2016) and the Elliptic2 paper (Bellei et al., 2024); training on innocent look-alikes follows Jullum et al. (2020). No code from these works is used.
- **Not used:** no pre-trained models, no external data or labels, and no outside services when making predictions.

## Licence

MIT (see `LICENSE`).
