# Model card — RunSight GRU

## Purpose
Predict, from the raw signal of a **single** running workout, whether it was spent mostly below (`aerobik`) or meaningfully above (`anaerobik`) the athlete's personal aerobic threshold, as defined by the rule-based label ([labelling method](labeling-method.md)). It lets the product label a first-ever workout without history.

## Not intended for
Medical or diagnostic use; absolute physiological claims; athletes or sports far from the training distribution (the data comes from recreational users of one fitness app).

## Data
EndomondoHR / FitRec, filtered to runs and passed through the QC rules; each workout is resampled to 500 points. Roughly 800 distinct users, with a very uneven number of workouts per user. The dataset is not redistributed.

## Input / output
* Input `(500, 4)`: speed (km/h), heart rate (bpm), altitude (m), GPS projected on its first principal axis. Each channel is min-max scaled with a scaler fitted on the training split (each time step is a feature, so inference must use the same `(1, 500)` shape — see `deploy/handler.py`).
* Output: sigmoid probability of "anaerobic"; decision threshold 0.5; `confidence = |p − 0.5| × 2`.

## Architecture and training
`GRU(units_1, return_sequences) → Dropout → GRU(units_2) → Dropout → Dense(1, sigmoid)`, Adam, binary cross-entropy, balanced class weights. Hyperparameters were selected with Keras Tuner `RandomSearch` (10 trials, objective `val_auc`, 30 epochs per trial; best trial 0.8464). Best: `units_1=64, units_2=16, dropout_1=0.1, dropout_2=0.5, l2_1=0, lr=1e-3`. The final model was trained for 50 epochs with `EarlyStopping(val_auc, patience=5)`, which never triggered; best validation AUC 0.8491 at epoch 47.

## Evaluation (test set, 13,884 workouts, per-workout split)

| | aerobic | anaerobic |
|---|---|---|
| precision | 0.78 | 0.75 |
| recall | 0.73 | 0.79 |
| support | 6,980 | 6,904 |

AUC 0.850, accuracy 0.76. Confusion matrix `[[5114, 1866], [1422, 5482]]` (rows: true aerobic / anaerobic). Raw numbers: [`results/test_metrics.json`](results/test_metrics.json).

### Error structure
Errors concentrate where the label itself is borderline: 45 % error for workouts with 0.01 < `anaerobic_time_frac` ≤ 0.02, 12–19 % for > 0.10. 41 % of workouts have `anaerobic_time_frac = 0` (error 23 %).

## Limitations
* **Fidelity to a rule, not accuracy against truth.** Labels are rule-generated.
* **Per-workout split.** Workouts of one athlete can be on both sides, so generalisation to *unseen athletes* is **unmeasured**.
* The model never sees the athlete's history, so it cannot, by construction, reproduce the personal threshold exactly; ~76 % agreement is the measured price of cold-start capability.
* Treat low-confidence outputs (< 0.6) as uncertain — the product routes those to the athlete.

## Serving
Container image on AWS Lambda ([`deploy/`](../deploy)); the model is loaded once per warm container. No inference-time QC: malformed streams return HTTP 400 (for example a missing `velocity_smooth`, `heartrate` or `latlng` key).
