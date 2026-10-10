# Explainable IDS: SHAP/LIME Cross-Dataset Reliability

This repository evaluates binary intrusion-detection models on aligned CICIDS2017 and UNSW-NB15 features and caches SHAP/LIME explanations for in-domain and cross-dataset experiments.

## Setup
1. `python -m pip install -r requirements.txt`
2. Run the smoke pipeline: `python run_all.py --smoke`
3. Run the full pipeline: `python run_all.py`

## Datasets
- CICIDS2017
- UNSW-NB15

## Project Structure
- `data/raw/` — place downloaded datasets here (see Datasets section)
- `data/preprocessed/aligned/` — committed aligned inputs and finalized representations
- `notebooks/` — analysis notebooks, one per pipeline phase
- `src/` — reusable Python modules (data loading, preprocessing, model, explain)
- `models/` — saved trained model weights (gitignored)
- `results/` — output plots and metric tables

## Experiments

- A: CICIDS to CICIDS test
- B: UNSW to UNSW test
- C: CICIDS to UNSW test
- D: UNSW to CICIDS test

## Limitations

- The committed aligned data contains train/test feature-row overlaps. Finalization quarantines overlapping test rows in new files; original inputs are preserved.
- Raw source tables and scaler artifacts are unavailable, so `raw`, `log`, and `scaled` finalized files preserve recovered aligned model-space values. Raw-unit explanation claims are not valid until raw data is restored.
- CICIDS represents one day with few attacks.
- Only nine shared features are available and several are strongly correlated.
- Source-vs-domain scaling is exposed, but source scaling cannot be reconstructed without the original scaler.
- Explanation sampling is class-balanced rather than prevalence-balanced.

## Results

Generated outputs are written under `results/`. Numeric interpretation is intentionally left to the report author.