"""Scoring wrapper (public window only) and label-free proxy checks.

The public window is used to check alignment and label meaning, never to choose parameters
(RULES.md, EVALUATION.md). Proxy checks run on the whole match and need no ground truth.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import numpy as np

from .features import runs
from .load import FPS, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
ABLATION = os.path.join(ROOT, "reports", "ablation.csv")


def score_public(labels: np.ndarray, bootstrap_n: int = 0) -> dict:
    """The organisers' metrics on the public window, via evaluation/possession_metrics.py."""
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    return pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=bootstrap_n)


def spells(labels: np.ndarray, graded: np.ndarray) -> list[tuple[str, int, int]]:
    """Possessions after the scorer's forward fill of X, per half: (team, start, end)."""
    out = []
    hs, he = runs(graded)
    for lo, hi in zip(hs, he):
        seg = labels[lo:hi].copy()
        last = None
        for i, x in enumerate(seg):  # forward fill X within the half
            if x == "X":
                seg[i] = last if last is not None else "X"
            else:
                last = x
        i = 0
        while i < len(seg):
            j = i
            while j < len(seg) and seg[j] == seg[i]:
                j += 1
            if seg[i] in ("H", "A"):
                out.append((seg[i], lo + i, lo + j))
            i = j
    return out


def proxies(labels: np.ndarray, graded: np.ndarray) -> dict:
    """Label-free sanity numbers for the whole match (docs/YT.md section 6)."""
    g = labels[graded]
    sp = spells(labels, graded)
    lengths = np.array([b - a for _, a, b in sp]) / FPS if sp else np.zeros(0)
    alive = np.isin(g, ["H", "A"])
    minutes = graded.sum() / FPS / 60
    return {
        "dead_share": float(np.mean(g == "D")),
        "x_share": float(np.mean(g == "X")),
        "n_possessions": len(sp),
        "possessions_per_min": len(sp) / minutes,
        "mean_possession_s": float(lengths.mean()) if len(lengths) else 0.0,
        "median_possession_s": float(np.median(lengths)) if len(lengths) else 0.0,
        "share_under_1s": float(np.mean(lengths < 1)) if len(lengths) else 0.0,
        "home_share_in_play": float(np.mean(g[alive] == "H")) if alive.any() else 0.0,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def log_ablation(name: str, settings: dict, public: dict, prox: dict, note: str = "") -> None:
    """Append one row to reports/ablation.csv."""
    os.makedirs(os.path.dirname(ABLATION), exist_ok=True)
    row = {
        "name": name, "commit": git_commit(), "settings": json.dumps(settings, sort_keys=True),
        "public_macro_f1": round(public["macro_f1"], 4),
        "public_spell_f1": round(public["spell_f1_mean"], 4),
        "public_n_pred_spells": public["n_pred_spells"],
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in prox.items()},
        "note": note,
    }
    new = not os.path.isfile(ABLATION)
    with open(ABLATION, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
