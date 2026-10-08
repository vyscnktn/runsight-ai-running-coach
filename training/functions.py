"""
EndomondoHR (FitRec) running workouts - aerobic / anaerobic labelling.

Data layout assumption: each row is one workout, each column a time series
resampled to 500 points (timestamp, speed, heart_rate, latitude, longitude,
altitude). timestamp is in Unix seconds.

Source: https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html

Pipeline (see run_zone_pipeline): raw CSV -> QC -> feature extraction ->
personal heart-rate-zone threshold -> aerobic/anaerobic label. No clustering
(GMM/K-means): long vs. easy runs are trivially separable by duration, and the
meaningful split (below/above-threshold effort, Seiler's polarized training
model) is computed directly from the personal heart-rate threshold.
"""

import ast
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------


def parse_arr(s):
    if isinstance(s, np.ndarray):
        return s.astype(np.float64)
    cleaned = s.replace("[", "").replace("]", "").replace(",", " ")
    return np.fromstring(cleaned, sep=" ", dtype=np.float64)



# ----------------------------------------------------------------------
# 2. Row-level quality control
# ----------------------------------------------------------------------

QC_RULES = {
    "min_points": 100,        # a series with fewer valid points is unreliable
    "max_nan_frac": 0.20,     # drop the workout if more than 20% of values are missing
    "min_duration_min": 10.0,
    "max_duration_min": 300.0,
    "speed_valid": (1.0, 30.0),      # km/h - anything outside is a GPS error
    "hr_valid": (60.0, 220.0),       # bpm
    "min_hr_frac": 0.50,      # at least half of the HR values must be in the valid range
}


def qc_row(row):
    """Evaluate one workout. Returns (is_valid, reason, clean_series)."""
    ts = row.get("timestamp")
    spd = row.get("speed")
    hr = row.get("heart_rate")


    duration_min = (ts[-1] - ts[0]) / 60.0
    if not np.isfinite(duration_min):
        return False, "bozuk_timestamp", None
    if duration_min < QC_RULES["min_duration_min"]:
        return False, "cok_kisa", None
    if duration_min > QC_RULES["max_duration_min"]:
        return False, "cok_uzun", None

    # Mark invalid values as NaN (instead of dropping them, so the time alignment is kept)
    lo, hi = QC_RULES["speed_valid"]
    spd = np.where((spd >= lo) & (spd <= hi) & np.isfinite(spd), spd, np.nan)
    lo, hi = QC_RULES["hr_valid"]
    hr = np.where((hr >= lo) & (hr <= hi) & np.isfinite(hr), hr, np.nan)

    spd_nan = np.isnan(spd).mean()
    hr_nan = np.isnan(hr).mean()
    if spd_nan > QC_RULES["max_nan_frac"]:
        return False, "hiz_cok_eksik", None
    if hr_nan > (1.0 - QC_RULES["min_hr_frac"]):
        return False, "hr_cok_eksik", None

    # Fill the remaining gaps linearly (time series: neighbouring points are a good estimate)
    spd = _interp_nan(spd)
    hr = _interp_nan(hr)

    return True, "ok", {"ts": ts, "speed": spd, "hr": hr,
                        "duration_min": duration_min,
                        "nan_frac": max(spd_nan, hr_nan)}


def _interp_nan(arr):
    """Fill NaNs by linear interpolation from neighbouring points."""
    nans = np.isnan(arr)

    idx = np.arange(len(arr))
    out = arr.copy()
    out[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return out

# ----------------------------------------------------------------------
# 3. Features
# ----------------------------------------------------------------------

def extract_features(series):
    """
    Extract duration-independent, interpretable features from one workout.

    Important: the 500 points are spread over the WHOLE workout, i.e. ~14 s apart
    for a 2-hour run and ~3.6 s apart for a 30-minute run. Differences between
    consecutive points therefore depend on duration and are not comparable
    across workouts. The measures below are either scale-free (CV, ratios)
    or normalised by time.
    """
    ts = series["ts"]
    spd = series["speed"]
    hr = series["hr"]
    dur = series["duration_min"]

    dt = np.diff(ts)                          # seconds
    dist_km = float(np.sum(spd[:-1] * dt) / 3600.0)

    med_spd = float(np.median(spd))
    p10, p90 = np.percentile(spd, [10, 90])

    # Pace variability - scale-free
    speed_cv = float(np.std(spd) / med_spd) if med_spd > 0 else 0.0
    speed_spread = float((p90 - p10) / med_spd) if med_spd > 0 else 0.0

    # Interval signature: how often did speed rise above 115% of its own median and drop back?
    # Normalised per minute -> independent of duration
    surges = _count_surges(spd, threshold=med_spd * 1.15)
    surge_rate = surges / dur if dur > 0 else 0.0

    # Long-run signature: how much slower is the last quarter than the first,
    # and how much did heart rate rise (cardiac drift)
    q = max(len(spd) // 4, 1)
    pace_drift = float(np.mean(spd[-q:]) / np.mean(spd[:q])) if np.mean(spd[:q]) > 0 else 1.0
    hr_drift = float(np.mean(hr[-q:]) - np.mean(hr[:q]))

    return {
        "duration_min": dur,
        "distance_km": dist_km,
        "speed_med": med_spd,
        "speed_cv": speed_cv,
        "speed_spread": speed_spread,
        "surge_rate": surge_rate,
        "pace_drift": pace_drift,
        "hr_med": float(np.median(hr)),
        "hr_p10": float(np.percentile(hr, 10)),
        "hr_p95": float(np.percentile(hr, 95)),
        "hr_drift": hr_drift,
        # metadata - not used as model features, only for filtering/reporting
        "_nan_frac": series["nan_frac"],
    }


def _count_surges(x, threshold, min_len=3):
    """Number of separate blocks where the signal stays above the threshold."""
    above = x > threshold
    if not above.any():
        return 0
    # Find block starts
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    ends = np.where(edges == -1)[0] + 1
    if above[-1]:
        ends = np.r_[ends, len(x)]
    return int(np.sum((ends - starts) >= min_len))


def _run_lengths_above(mask, min_len=3):
    """
    List of lengths (in samples) of the separate blocks above the threshold in a boolean mask.
    A tempo run produces ONE LONG block, intervals produce MANY SHORT blocks -
    add_hr_zone_features uses this distinction.
    """
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    ends = np.where(edges == -1)[0] + 1
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    lengths = ends - starts
    return lengths[lengths >= min_len].tolist()


# ----------------------------------------------------------------------
# 5. Per-user normalisation
# ----------------------------------------------------------------------

PERSONAL_COLS = ["speed_med", "hr_med", "hr_p95"]


def add_personal_features(X, user_ids, min_workouts=5):
    """
    The same 11 km/h is an easy pace for a good runner and race pace for a beginner.
    Absolute speed/heart rate are not comparable across people.
    Every measure is divided by the person's own median.

    For people with too few workouts the global median is used and the row is
    flagged with _personal_ref=False.
    """
    X = X.copy()
    X["_user"] = np.asarray(user_ids)

    counts = X["_user"].value_counts()
    enough = set(counts[counts >= min_workouts].index)
    X["_personal_ref"] = X["_user"].isin(enough)

    for col in PERSONAL_COLS:
        global_ref = X[col].median()
        user_ref = X.groupby("_user")[col].transform("median")
        ref = np.where(X["_personal_ref"], user_ref, global_ref)
        ref = np.where(np.asarray(ref) > 0, ref, global_ref)
        X[f"rel_{col}"] = X[col] / ref

    return X


# ----------------------------------------------------------------------
# 5b. Personalised heart-rate-zone features
# ----------------------------------------------------------------------
#
# Why here: there is no age/HRmax formula (220 - age etc.) because the dataset has no
# age information, and the lactate threshold cannot be known in absolute terms. But we
# can use the heart-rate range observed across ALL of a person's workouts as the reference
# - a data-driven, personalised version of Karvonen's heart-rate-reserve idea. This turns
# the missing age data from a disadvantage into an advantage: instead of a population-average
# formula, we use the person's own real
# effort history.
#
# Distinguishing logic:
#   - easy/long run: almost no time above the threshold
#   - tempo/threshold: ONE LONG block above the threshold (sustained effort)
#   - interval: MANY SHORT blocks above the threshold (repeated effort)

def add_hr_zone_features(X, series_list, threshold_frac=0.70, min_workouts=5):
    """
    Add heart-rate-zone features to X. series_list must be in the SAME ORDER as X
    (it is called right after add_personal_features inside prepare_features,
    while series_list is still in memory).

    threshold_frac=0.70: approximately the AEROBIC THRESHOLD (VT1) - the right boundary
    for the binary below/above-threshold split (Seiler's polarized training model).
    A higher value (e.g. 0.85) corresponds to the LACTATE THRESHOLD (VT2) and only
    catches interval-level peaks - sustained tempo/threshold efforts (typically
    70-80% HRR) would fall below it and be wrongly labelled
    "aerobic" (this exact failure showed up in synthetic scenario tests; see
    tests/test_labeling.py for the scenarios).

    personal_hr_ceiling: the highest p95 heart rate across ALL of the person's workouts
                         - proxy for "the hardest effort this person shows in the data"
    personal_hr_floor:   the lowest p10 heart rate across ALL of the person's workouts
                         - proxy for "close-to-resting effort for this person"
    threshold = floor + threshold_frac * (ceiling - floor)
    """
    X = X.copy()

    ceiling = X.groupby("_user")["hr_p95"].transform("max")
    floor = X.groupby("_user")["hr_p10"].transform("min")

    counts = X["_user"].value_counts()
    enough = X["_user"].isin(counts[counts >= min_workouts].index)
    global_ceiling = X["hr_p95"].max()
    global_floor = X["hr_p10"].min()
    ceiling = np.where(enough, ceiling, global_ceiling)
    floor = np.where(enough, floor, global_floor)

    hr_range = np.maximum(ceiling - floor, 1e-6)
    threshold = floor + threshold_frac * hr_range
    X["hr_reserve_frac"] = np.clip((X["hr_p95"].to_numpy() - floor) / hr_range, 0, 1)

    time_above, excursion_rate, max_excursion_min = [], [], []
    for pos, (thr, s) in enumerate(zip(threshold, series_list)):
        hr = s["hr"]
        above = hr > thr
        n_samples = len(hr)
        dur = s["duration_min"]

        time_above.append(float(above.mean()))

        lengths = _run_lengths_above(above, min_len=3)
        excursion_rate.append(len(lengths) / dur if dur > 0 else 0.0)
        if lengths:
            max_len_frac = max(lengths) / n_samples
            max_excursion_min.append(max_len_frac * dur)
        else:
            max_excursion_min.append(0.0)

    X["hr_time_above_threshold"] = time_above
    X["hr_excursion_rate"] = excursion_rate
    X["hr_max_excursion_min"] = max_excursion_min

    return X


# ----------------------------------------------------------------------
# 6. Shared preparation - after loading, before labelling
# ----------------------------------------------------------------------

def prepare_features(df, user_col="userId", min_n=20, verbose=True):
    """
    Raw df -> feature matrix X + series_list (raw, cleaned time series).

    Steps: sport filter -> QC -> feature extraction -> personal
    normalisation -> heart-rate-zone features.
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"Raw data: {len(df)} workouts")


    log(f"  workouts after sport filter: {len(df)}")

    idx, series_list, reasons = [], [], []
    for i, row in df.iterrows():
        ok, reason, s = qc_row(row)
        reasons.append(reason)
        if ok:
            idx.append(i)
            series_list.append(s)
    log("  QC:", pd.Series(reasons).value_counts().to_dict())

    if len(idx) < min_n:
        raise ValueError(f"Too few workouts left: {len(idx)}")

    X = pd.DataFrame([extract_features(s) for s in series_list], index=idx)
    users = df.loc[idx, user_col].to_numpy() if user_col in df.columns else np.zeros(len(idx))
    X = add_personal_features(X, users)

    X = add_hr_zone_features(X, series_list)   # must be in the SAME ORDER as series_list
    log(f"  feature matrix: {X.shape}")
    log(f"  share with a personal reference: {X['_personal_ref'].mean():.0%}")

    return X, series_list, idx


# ----------------------------------------------------------------------
# 7. Aerobic / anaerobic - directly computed binary label
# ----------------------------------------------------------------------
#
# Long run vs. easy run is trivially separable by LOOKING at duration -
# no clustering needed. The meaningful split is below/above-threshold
# effort (Seiler's polarized training model), and we already have a
# physiologically grounded signal for it: hr_time_above_threshold
# (fraction of time above the threshold derived from the person's own heart-rate history).

def add_training_zone(X, cutoff=0.02):
    """
    Adds the binary 'aerobik' / 'anaerobik' label plus a continuous score.

    anaerobic_time_frac: alias of hr_time_above_threshold - the fraction of the
    workout spent above the personal threshold. KEEP it as a continuous score:
    for mixed workouts such as "sprints during an easy run" it carries the real
    proportion instead of forcing a single class (0.05 = 95% aerobic + 5%
    anaerobic - you do not have to call it either class,
    but you can still convert it to a binary decision with cutoff).

    cutoff: in synthetic tests pure easy/long runs came out at exactly 0.0 (see
    tests/test_labeling.py); 0.02 is a small margin against noise-induced
    false positives. Inspect the distribution with `training_zone_histogram`
    on your own data and tune it to where the pile-up at zero ends.
    """
    X = X.copy()
    X["anaerobic_time_frac"] = X["hr_time_above_threshold"]
    X["training_zone"] = np.where(X["anaerobic_time_frac"] > cutoff,
                                  "anaerobik", "aerobik")
    return X


def run_zone_pipeline(df, cutoff=0.02, user_col="userId", verbose=True):
    """
    Raw df -> anaerobic_time_frac + training_zone for every workout.

    Steps: QC + personal heart-rate-zone threshold + threshold application.
    No clustering (GMM/K-means / choosing K).
    """
    X, series_list, idx = prepare_features(df, user_col=user_col, verbose=verbose)
    X = add_training_zone(X, cutoff=cutoff)

    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    counts = X["training_zone"].value_counts()
    log(f"  training_zone: {counts.to_dict()}")

    return X, {"series_list": series_list, "index": idx, "cutoff": cutoff}
