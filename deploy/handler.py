"""
AWS Lambda handler: aerobic/anaerobic prediction from a Strava activity stream.

Input (event["body"], JSON) - the Strava response of
GET /activities/{id}/streams?keys=heartrate,altitude,latlng,velocity_smooth
exactly as returned:

    {
      "heartrate":        {"data": [bpm, bpm, ...]},
      "altitude":         {"data": [m, m, ...]},
      "latlng":           {"data": [[lat, lon], [lat, lon], ...]},
      "velocity_smooth":  {"data": [m/s, m/s, ...]}
    }

n8n only has to POST the streams response it got from the Strava API to
this endpoint unchanged.

Output (JSON):
    {
      "training_zone": "aerobik" | "anaerobik",
      "anaerobic_probability": float,   # sigmoid output of the model (0-1)
      "confidence": float,              # |probability - 0.5| * 2  (0-1)
      "hr_p10": float | null,           # 10th percentile of this workout's HR (bpm)
      "hr_p95": float | null            # 95th percentile of this workout's HR (bpm)
    }

The labels "aerobik" / "anaerobik" are the API contract with the n8n
workflow and the agent prompts, so they are intentionally not translated.

hr_p10 / hr_p95 use the SAME method as functions.py::add_hr_zone_features
(np.percentile(hr, 10) / np.percentile(hr, 95)). They are exposed so that n8n
can log them per workout and periodically recompute the personal threshold
(threshold = floor + 0.70 * (ceiling - floor), with
personal_hr_ceiling = max(hr_p95) and personal_hr_floor = min(hr_p10) over ALL
of the user's workouts), i.e. automatic calibration. The model and its
architecture are unchanged; an intermediate value that was already computed is
simply returned.

The following two files must sit next to this file (inside the same Docker
image):
    gru_model.keras         (model produced by 6_gru_preprocess + 7_gru_tuner)
    gru_preprocessing.pkl   (channel order + scalers fitted during training)
"""

import json
import pickle

import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "gru_model.keras"
PREPROC_PATH = "gru_preprocessing.pkl"
N_POINTS = 500          # fixed sequence length used in training
R_EARTH = 6371000.0     # metres - for the lat/lon -> local coordinate conversion

# Loaded once per container, as long as the Lambda container stays warm
# (warm invocation), not on every call - adds no latency apart from the cold start.
_model = load_model(MODEL_PATH)
with open(PREPROC_PATH, "rb") as f:
    _preproc = pickle.load(f)
_channel_order = _preproc["channel_order"]
_scalers = _preproc["scalers"]


def _resample_to_n(values, n=N_POINTS):
    """Resample a Strava stream (variable length) to a fixed n points, as in training."""
    values = np.asarray(values, dtype=np.float64)
    old_idx = np.linspace(0.0, 1.0, num=len(values))
    new_idx = np.linspace(0.0, 1.0, num=n)
    return np.interp(new_idx, old_idx, values)


def _interp_nan(arr):
    """Fill NaNs left over from sensor dropouts (HR loss etc.) from their neighbours."""
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.arange(len(arr))
        arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])
    return arr


def _latlon_to_pca(lat, lon):
    """Same transform as in training (6_gru_preprocess.ipynb): first convert to
    local metre coordinates (equirectangular approximation), then project onto
    the single axis of highest variance (PCA, 1 component)."""
    lat0_rad = np.radians(lat[0])
    x = R_EARTH * np.radians(lon - lon[0]) * np.cos(lat0_rad)
    y = R_EARTH * np.radians(lat - lat[0])

    xy = np.column_stack([x, y])
    xy = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    return xy @ vt[0]


def _hr_percentiles(hr_raw):
    """SAME method as functions.py::add_hr_zone_features: the 10th/95th
    percentile of this workout's heart rate. NaNs are left out; if there is no
    valid value, returns (None, None)."""
    hr_raw = np.asarray(hr_raw, dtype=np.float64)
    hr_raw = hr_raw[np.isfinite(hr_raw)]
    if hr_raw.size == 0:
        return None, None
    p10 = float(np.percentile(hr_raw, 10))
    p95 = float(np.percentile(hr_raw, 95))
    return p10, p95


def _extract_channels(streams):
    """Extract the 4 training channels (speed, heart_rate, altitude,
    latlon_pca) from the Strava stream JSON and resample each to 500 points."""
    if "velocity_smooth" not in streams:
        raise ValueError(
            "velocity_smooth stream is missing - add the 'velocity_smooth' key "
            "to the Strava streams request (Strava computes it automatically "
            "for activities with GPS)."
        )

    hr = np.asarray(streams["heartrate"]["data"], dtype=np.float64)
    alt = np.asarray(streams["altitude"]["data"], dtype=np.float64)
    latlng = np.asarray(streams["latlng"]["data"], dtype=np.float64)
    lat, lon = latlng[:, 0], latlng[:, 1]
    speed_kmh = np.asarray(streams["velocity_smooth"]["data"], dtype=np.float64) * 3.6

    raw = {
        "speed": speed_kmh,
        "heart_rate": hr,
        "altitude": alt,
        "latlon_pca": _latlon_to_pca(lat, lon),
    }

    channels = [
        _interp_nan(_resample_to_n(raw[name]))
        for name in _channel_order
    ]
    return np.column_stack(channels)  # shape: (500, n_channels)


def _scale_channels(x):
    """Apply the MinMaxScalers saved during training. In training the scalers
    were fitted with each time step (500 of them) as a separate 'feature', so
    here each channel is passed with shape (1, 500) as well - this must match
    the fit shape exactly, otherwise the scaler is misaligned."""
    x_scaled = np.empty_like(x)
    for c, scaler in enumerate(_scalers):
        col = x[:, c].reshape(1, -1)
        x_scaled[:, c] = scaler.transform(col).ravel()
    return x_scaled


def lambda_handler(event, context):
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)
        # When n8n exports a node's output (Download JSON) it always wraps it
        # in an item list ([{...}]) - that is n8n's internal data format.
        # If a single-element list arrives, unwrap the dict inside.
        if isinstance(body, list):
            if len(body) != 1:
                raise ValueError(f"Expected a single activity, got {len(body)} items")
            body = body[0]

        hr_p10, hr_p95 = _hr_percentiles(body.get("heartrate", {}).get("data", []))

        x = _extract_channels(body)
        x_scaled = _scale_channels(x)
        x_batch = x_scaled[np.newaxis, ...]  # (1, 500, n_channels)

        proba = float(_model.predict(x_batch, verbose=0)[0, 0])
        zone = "anaerobik" if proba >= 0.5 else "aerobik"
        confidence = abs(proba - 0.5) * 2

        return {
            "statusCode": 200,
            "body": json.dumps({
                "training_zone": zone,
                "anaerobic_probability": round(proba, 4),
                "confidence": round(confidence, 4),
                "hr_p10": round(hr_p10, 1) if hr_p10 is not None else None,
                "hr_p95": round(hr_p95, 1) if hr_p95 is not None else None,
            }),
        }
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    # Quick local test before deploying to AWS. Usage:
    #   python handler.py path/to/streams.json
    import sys
    with open(sys.argv[1] if len(sys.argv) > 1 else "sample_streams.json") as f:
        sample = json.load(f)
    print(lambda_handler({"body": json.dumps(sample)}, None))
