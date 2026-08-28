# Explainable IDS: SHAP/LIME Cross-Dataset Reliability

Cross-dataset evaluation of a DL-based intrusion detection system 
(CICIDS2017 + UNSW-NB15), comparing SHAP/LIME explanation reliability 
across in-distribution vs cross-dataset predictions.

## Setup
1. Clone the repo
2. `pip install -r requirements.txt`
3. Download datasets (see below) into `data/raw/`
4. Run notebooks in order from `notebooks/`

## Datasets
- CICIDS2017
- UNSW-NB15

## Project Structure
- `data/raw/` — place downloaded datasets here (see Datasets section)
- `data/processed/` — cleaned/processed data (auto-generated)
- `notebooks/` — analysis notebooks, one per pipeline phase
- `src/` — reusable Python modules (data loading, preprocessing, model, explain)
- `models/` — saved trained model weights (gitignored)
- `results/` — output plots and metric tables