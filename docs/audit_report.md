# Phase 0 Audit Report

## Scope

This audit inspected `README.md`, `requirements.txt`, `.gitignore`, the files in `src/`, the preprocessing notebooks, and the four aligned CSV files under `data/preprocessed/aligned/`. The repository currently contains no model, training, evaluation, XAI, or test implementation.

## Verified aligned-data findings

- All four aligned files have the same nine feature columns, in the same order, followed by `label`: `flow_duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`, `fwd_to_bwd_pkt_ratio`, `fwd_to_bwd_byte_ratio`, `bytes_per_packet`, and `packets_per_second`.
- Labels are numeric binary values `{0, 1}`. Current counts are:
  - CICIDS train: 129,726 benign / 1,714 attack.
  - CICIDS test: 32,431 benign / 429 attack.
  - UNSW train: 56,000 benign / 119,341 attack.
  - UNSW test: 37,000 benign / 45,332 attack.
- Separate train/test files exist. The CICIDS notebook output documents an 80/20 random split and scaled aligned files (`notebooks/data_preprocssing_cicds.ipynb`, final validation/save cells). The UNSW notebook writes official-looking train/test filenames, but the split provenance is not independently recoverable from the aligned files alone.
- The aligned tables are already scaled: the CICIDS notebook explicitly applies `StandardScaler` and saves scaled aligned tables. The corresponding scaler artifact is not present under the repository tree inspected. No raw aligned copy is present.
- No missing or infinite values were found in any aligned file.
- Skewness is high for every feature in at least one split. In particular, both ratio features exceed skewness 2 in every reported split. The existing preprocessing notebook does not document a `log1p` transform before scaling.
- Exact full-row train/test overlap exists: CICIDS 3,525 rows and UNSW-NB15 1,905 rows. Feature-only overlap is 3,512 and 2,037 rows respectively. This violates the no-leakage requirement and is a stop condition.
- The ratio behavior cannot be conclusively reconstructed from the aligned files. Existing preprocessing code (`src/preprocessing.py`, `handle_invalid_values`) replaces infinities with missing values and imputes medians; the feature-engineering cells that define the ratios were not preserved in a reusable source module.
- Byte semantics cannot be conclusively established for both datasets from the current reusable source. CICIDS notebook cells derive byte features from CICIDS flow columns; UNSW notebook cells use `sbytes`/`dbytes`, but the aligned derivation contract is not recorded in `src/`.
- CICIDS notebook output identifies one Thursday morning web-attack file and the three attack labels Brute Force, XSS, and SQL Injection. The aligned files retain no timestamp column.

## Domain shift check

A five-fold logistic-regression classifier trained to distinguish CICIDS from UNSW rows using the nine aligned features produced mean ROC-AUC `0.4962`, i.e. chance-level for this particular scaled representation. Per-feature KS statistics were: `flow_duration 0.838`, `fwd_packets 0.812`, `bwd_packets 0.805`, `fwd_bytes 0.766`, `bwd_bytes 0.914`, `fwd_to_bwd_pkt_ratio 0.390`, `fwd_to_bwd_byte_ratio 0.784`, `bytes_per_packet 0.289`, `packets_per_second 0.549`. The low linear AUC does not remove the strong univariate distribution differences.

## Repository inconsistencies

- `README.md` says `data/processed/`, while the actual data is under `data/preprocessed/aligned/`.
- `src/preprocessing.py` uses hardcoded relative paths for an absent `data/raw/` file and writes the stale `data/CICDS_cleaned.csv` layout.
- The notebooks contain hardcoded absolute temporary paths in recorded outputs and use notebook-local preprocessing logic.
- `requirements.txt` omits `joblib`, `scipy`, `pyyaml`, `tqdm`, and `pytest` requested by the pipeline.
- `.gitignore` does not ignore `models/` or `results/xai_cache/`.
- `src/model.py`, `src/explain.py`, and `src/data_loader.py` do not implement the requested pipeline.

## Changes needed before training

1. Resolve the train/test duplicate-row leakage and regenerate or explicitly quarantine the affected rows using a documented policy. Training must stop until this is decided.
2. Recover raw feature data or document that raw/log reconstruction is impossible from the currently committed artifacts; the current scaled tables are insufficient for a faithful raw-unit explanation pipeline.
3. Rebuild a single source-of-truth preprocessing configuration and save train-fitted log/scaler artifacts.
4. Record the exact ratio and byte-feature definitions for both datasets.
5. Replace stale paths and notebook-side logic with importable `src/` implementations, then add tests before training.

## Post-audit continuation policy

The user authorized continuation without recovering raw source files. Phase 1 quarantines every test row whose nine-feature tuple occurs in that dataset's train split, producing new finalized files while preserving the original aligned CSVs. Because raw units, the original log transform, and the original scaler cannot be recovered, the finalized `raw`, `log`, and `scaled` files are explicitly marked as recovered aligned model-space values; no raw-unit interpretation should be made from them.