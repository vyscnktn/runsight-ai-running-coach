#!/usr/bin/env python3
"""Regenerate the two README figures that do not need the model or the test set.

    python scripts/make_figures.py [--labels training/data/aerobik_anaerobik.csv]

* docs/images/learning_curve.png          from docs/results/gru_history_original.json
* docs/images/confusion_matrix.png        from docs/results/test_metrics.json
* docs/images/training_zone_histogram.png from the labelled dataset (optional)

ROC curve, confusion matrix and the error-by-threshold-distance plot are
produced by training/9_gru_evaluate.ipynb.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "images")
ORANGE, BLUE = "#f26b21", "steelblue"


def learning_curve():
    h = json.load(open(os.path.join(ROOT, "docs", "results", "gru_history_original.json")))
    ep = range(1, len(h["loss"]) + 1)
    best = max(range(len(h["val_auc"])), key=lambda i: h["val_auc"][i])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tr, va, name) in zip(axes, [("loss", "val_loss", "Loss"), ("auc", "val_auc", "AUC")]):
        ax.plot(ep, h[tr], color=BLUE, label="train")
        ax.plot(ep, h[va], color=ORANGE, label="validation")
        ax.axvline(best + 1, ls="--", color="gray", lw=1, label=f"best val AUC (epoch {best + 1})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{name} per epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "learning_curve.png"), dpi=150)
    plt.close(fig)
    print(f"learning_curve.png (best val_auc {h['val_auc'][best]:.4f} at epoch {best + 1})")


def confusion():
    m = json.load(open(os.path.join(ROOT, "docs", "results", "test_metrics.json")))
    cm = m["confusion_matrix"]["rows_true_cols_pred"]
    labels = m["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Oranges")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (rule-based label)")
    ax.set_title(f"Confusion matrix - test set (AUC {m['auc']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("confusion_matrix.png")


def histogram(path):
    import pandas as pd
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["anaerobic_time_frac"], bins=40, color=BLUE, edgecolor="white")
    ax.axvline(0.02, color=ORANGE, ls="--", label="cutoff = 0.02")
    ax.set_yscale("log")
    ax.set_xlabel("anaerobic_time_frac (fraction of time above the personal threshold)")
    ax.set_ylabel("Workouts (log scale)")
    ax.set_title("Distribution of anaerobic_time_frac and the label cutoff")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "training_zone_histogram.png"), dpi=150)
    plt.close(fig)
    print("training_zone_histogram.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(ROOT, "training", "data", "aerobik_anaerobik.csv"))
    a = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    learning_curve()
    confusion()
    if os.path.exists(a.labels):
        histogram(a.labels)
    else:
        print("labels CSV not found - skipped histogram")
