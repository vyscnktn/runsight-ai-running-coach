# Training pipeline

Notebooks are meant to be run in order. Outputs are cleared; each notebook starts with a header cell describing its inputs and outputs.

| # | Notebook | What it does | Writes |
|---|---|---|---|
| 1 | `1_data_preprocess` | Parses the raw EndomondoHR JSON Lines (Python-dict syntax, so `ast.literal_eval`, not `json`) into a CSV in batches | `data/endomondoHR_proper.csv` |
| 2 | `2_drop_columns` | Drops unused columns | `data/endomondoHR_proper.csv` |
| 3 | `3_haversine_savgol_filter` | Computes speed from GPS (haversine) with Savitzky–Golay smoothing for activities without speed | `data/endomondoHR_speed.csv` |
| 4 | `4_Kmeans` | Early clustering experiment (kept for the record — dropped, see [docs/labeling-method.md](../docs/labeling-method.md)) | – |
| 5 | `5_polarize` | Rule-based aerobic/anaerobic labelling, cutoff histogram, visual checks | `data/aerobik_anaerobik.csv` |
| 6 | `6_gru_preprocess` | Resamples to 500 points, builds the 4 channels, splits, fits scalers | `artifacts/gru_arrays.npz`, `artifacts/gru_preprocessing.pkl` |
| 7 | `7_gru_tuner` | Keras Tuner search and final training | `artifacts/gru_model.keras` |
| 9 | `9_gru_evaluate` | ROC/AUC, confusion matrix, error analysis against the label threshold | `../docs/images/*.png` |

`functions.py` holds the QC rules, feature extraction and the rule-based pipeline (`run_zone_pipeline`) and is covered by [`tests/`](../tests).

## Data

Download EndomondoHR / FitRec from <https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html> (not redistributed; check its terms) and place the raw file in `data/`. The raw files are several GB; notebook 1 processes them in batches (it was originally run on Colab because of RAM limits).

## Environment

```bash
pip install -r requirements.txt
```

After training, copy `artifacts/gru_model.keras` and `artifacts/gru_preprocessing.pkl` to [`../deploy/`](../deploy). Large artifacts (`*.npz`, `data/`, `artifacts/`) are git-ignored.

## Reproducing the numbers

Seeds are fixed (`46`) in notebook 7, but TensorFlow results can still differ slightly across hardware/versions. The reported run is stored in [`../docs/results/`](../docs/results) (`gru_history_original.json`, `test_metrics.json`).
