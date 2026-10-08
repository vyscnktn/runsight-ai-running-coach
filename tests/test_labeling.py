"""Scenario tests for the rule-based labelling pipeline (training/functions.py).

They use synthetic workouts with known structure - the same kind of scenario
tests that revealed why the threshold is 0.70 and not 0.85 of the personal
heart-rate range:

  * easy run      -> flat, low heart rate            -> must be "aerobik"
  * tempo run     -> one long sustained block        -> must be "anaerobik"
  * interval run  -> many short hard repetitions     -> must be "anaerobik"

Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import functions as fc  # noqa: E402

N = 500                       # points per workout
TS = np.arange(N) * 7.2       # 60 minutes


def _hr(kind, rng):
    noise = rng.normal(0, 1.5, N)
    if kind == "easy":
        return 124 + noise
    if kind == "tempo":              # long block at ~165 bpm from 20% to 85% of the run
        hr = 128 + noise
        hr[100:425] = 165 + noise[100:425]
        return hr
    if kind == "interval":           # 20 repetitions of 10 hard samples (~72 s) at ~176 bpm
        hr = 128 + noise
        for k in range(20):
            hr[60 + 20 * k: 70 + 20 * k] = 176 + noise[60 + 20 * k: 70 + 20 * k]
        return hr
    raise ValueError(kind)


def make_df(n_per_kind=8, seed=0):
    rng = np.random.default_rng(seed)
    rows, kinds = [], []
    for kind in ("easy", "tempo", "interval"):
        for _ in range(n_per_kind):
            rows.append({
                "userId": 1,
                "timestamp": TS,
                "speed": 10 + rng.normal(0, 0.2, N),
                "heart_rate": _hr(kind, rng),
            })
            kinds.append(kind)
    return pd.DataFrame(rows), np.array(kinds)


class TestQC(unittest.TestCase):
    def base(self):
        return {"timestamp": TS.copy(), "speed": np.full(N, 10.0), "heart_rate": np.full(N, 140.0)}

    def test_valid_workout_passes(self):
        ok, reason, _ = fc.qc_row(self.base())
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_too_short(self):
        r = self.base()
        r["timestamp"] = np.arange(N) * 0.5          # ~4 minutes
        self.assertEqual(fc.qc_row(r)[1], "cok_kisa")

    def test_too_long(self):
        r = self.base()
        r["timestamp"] = np.arange(N) * 60.0         # ~8 hours
        self.assertEqual(fc.qc_row(r)[1], "cok_uzun")

    def test_broken_timestamp(self):
        r = self.base()
        r["timestamp"] = np.full(N, np.nan)
        self.assertEqual(fc.qc_row(r)[1], "bozuk_timestamp")

    def test_missing_speed(self):
        r = self.base()
        r["speed"][: N // 2] = np.nan
        self.assertEqual(fc.qc_row(r)[1], "hiz_cok_eksik")

    def test_missing_heart_rate(self):
        r = self.base()
        r["heart_rate"][: int(N * 0.6)] = np.nan
        self.assertEqual(fc.qc_row(r)[1], "hr_cok_eksik")


class TestZoneLabelling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.kinds = make_df()
        cls.X, cls.info = fc.run_zone_pipeline(cls.df, verbose=False)

    def test_all_workouts_survive_qc(self):
        self.assertEqual(len(self.X), len(self.df))

    def test_easy_runs_are_aerobic_and_exactly_zero(self):
        easy = self.X[self.kinds == "easy"]
        self.assertTrue((easy["training_zone"] == "aerobik").all())
        self.assertTrue((easy["anaerobic_time_frac"] == 0.0).all())

    def test_tempo_runs_are_anaerobic(self):
        tempo = self.X[self.kinds == "tempo"]
        self.assertTrue((tempo["training_zone"] == "anaerobik").all())
        self.assertGreater(tempo["anaerobic_time_frac"].min(), 0.5)

    def test_interval_runs_are_anaerobic_with_many_short_excursions(self):
        iv = self.X[self.kinds == "interval"]
        tempo = self.X[self.kinds == "tempo"]
        self.assertTrue((iv["training_zone"] == "anaerobik").all())
        self.assertGreater(iv["hr_excursion_rate"].mean(), 5 * tempo["hr_excursion_rate"].mean())

    def test_threshold_070_catches_tempo_but_085_does_not(self):
        """Why 0.70: a sustained tempo effort sits between VT1 and VT2."""
        X, series_list, _ = fc.prepare_features(self.df, verbose=False)
        tempo_pos = np.where(self.kinds == "tempo")[0]
        for frac, expect_caught in ((0.70, True), (0.85, False)):
            Xz = fc.add_hr_zone_features(X.drop(columns=[
                "hr_time_above_threshold", "hr_excursion_rate", "hr_max_excursion_min"]),
                series_list, threshold_frac=frac)
            labelled = fc.add_training_zone(Xz)
            caught = (labelled.iloc[tempo_pos]["training_zone"] == "anaerobik").all()
            self.assertEqual(caught, expect_caught, f"threshold_frac={frac}")

    def test_cutoff_is_strict_greater_than(self):
        X = pd.DataFrame({"hr_time_above_threshold": [0.0, 0.02, 0.0201]})
        z = fc.add_training_zone(X, cutoff=0.02)["training_zone"].tolist()
        self.assertEqual(z, ["aerobik", "aerobik", "anaerobik"])


class TestSanitizer(unittest.TestCase):
    def test_sanitizer_removes_identifiers(self):
        import json
        import sanitize_workflow as sw
        wf = {
            "name": "t", "meta": {"instanceId": "abc"},
            "nodes": [{"name": "n", "webhookId": "w", "credentials": {"x": {"id": "123", "name": "N"}},
                       "parameters": {"url": "https://abcdefghij1234567890.lambda-url.eu-central-1.on.aws/",
                                      "mail": "someone@gmail.com"}}],
            "connections": {}, "pinData": {"n": [{"json": {"gps": 1}}]},
        }
        out = sw.sanitize(json.dumps(wf))
        d = json.loads(out)
        self.assertNotIn("pinData", d)
        self.assertNotIn("webhookId", d["nodes"][0])
        self.assertEqual(d["nodes"][0]["credentials"]["x"]["id"], "REPLACE_ME")
        self.assertNotIn("gmail.com", out)
        self.assertNotIn("abcdefghij", out)
        self.assertEqual(sw.leftovers(out), {})


if __name__ == "__main__":
    unittest.main()
