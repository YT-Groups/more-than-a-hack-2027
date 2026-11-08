"""Targeted restart audit for corners, free kicks, goal kicks, and kickoffs."""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pipeline import PipelineConfig, build  # noqa: E402
from yt.load import load_match  # noqa: E402
from yt.sanity import find_dead_runs, overlaps, read_submission  # noqa: E402
from yt import tags  # noqa: E402


TARGET_TYPES = {"corner kick", "free kick", "goal kick", "kick off (start)",
                "kick off (after goal)"}


def read_truth(path: str) -> np.ndarray:
    labels, _ = read_submission(path)
    return labels


def first_alive_after(labels: np.ndarray, restart: int, hi: int) -> int | None:
    for frame in range(max(0, restart), min(len(labels), hi)):
        if labels[frame] in ("H", "A"):
            return frame
    return None


def nearest_dead_run(truth_range: tuple[int, int],
                     predicted_runs: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not predicted_runs:
        return None
    truth_a, truth_b = truth_range

    def score(run):
        a, b = run
        overlap = max(0, min(truth_b, b) - max(truth_a, a))
        end_distance = abs(b - truth_b)
        start_distance = abs(a - truth_a)
        return (-overlap, end_distance + start_distance, abs(b - truth_b))

    return min(predicted_runs, key=score)


def alive_edge(labels: np.ndarray, start: int, end: int) -> tuple[int | None, str | None]:
    for frame in range(start, end):
        if labels[frame] == "D":
            continue
        if labels[frame] in ("H", "A"):
            return frame, str(labels[frame])
    return None, None


def audit(csv_path: str, truth_path: str | None = None) -> list[dict]:
    labels, frames = read_submission(csv_path)
    predicted_runs = find_dead_runs(labels)
    tag_rows = tags.load()
    rows = []

    truth = read_truth(truth_path) if truth_path else None
    truth_runs = find_dead_runs(truth) if truth is not None else []
    model_result = None
    try:
        _, model_result = build(PipelineConfig(), m=load_match())
    except Exception:
        model_result = None
    restart_team = model_result["restart_team"] if model_result else {}

    for tag in tag_rows:
        if tag["code"].strip().upper() != "SET PIECES":
            continue
        tag_type = (tag["tags"].get("Type") or "").strip().lower()
        if tag_type not in TARGET_TYPES:
            continue

        a = max(0, tag["start_frame"])
        b = min(len(labels), tag["end_frame"])
        team_hint = tags.TEAM.get((tag["tags"].get("Team") or "").strip().lower())

        truth_matches = []
        if truth is not None:
            truth_matches = [
                (ta, tb) for ta, tb in truth_runs
                if overlaps(a, b, ta, tb)
            ]
        truth_range = min(
            truth_matches,
            key=lambda run: max(0, min(b, run[1]) - max(a, run[0])) * -1,
            default=None,
        )

        matched = nearest_dead_run(truth_range, predicted_runs) if truth_range else None
        if matched is None:
            matched = min(
                (run for run in predicted_runs if overlaps(a, b, run[0], run[1])),
                key=lambda run: abs(run[0] - a),
                default=None,
            )

        truth_restart = truth_range[1] if truth_range else None
        predicted_restart = matched[1] if matched else None
        truth_team = str(truth[truth_restart]) if (
            truth_restart is not None and 0 <= truth_restart < len(truth)
        ) else None
        predicted_team = alive_edge(labels, matched[1], min(len(labels), matched[1] + 25))[1] \
            if matched else None
        model_team_value = restart_team.get(predicted_restart) if predicted_restart else None
        model_team = "H" if model_team_value == 1 else "A" if model_team_value == -1 else ""

        latency = predicted_restart - truth_restart if (
            predicted_restart is not None and truth_restart is not None) else None
        rows.append({
            "type": tag_type,
            "team_hint": "H" if team_hint == 1 else "A" if team_hint == -1 else "",
            "tag_range": "[{}, {})".format(a, b),
            "truth_dead_range": "[{}, {})".format(*truth_range) if truth_range else "",
            "predicted_dead_range": "[{}, {})".format(*matched) if matched else "",
            "truth_restart": truth_restart,
            "predicted_restart": predicted_restart,
            "restart_latency_frames": latency,
            "restart_latency_seconds": latency / 25.0 if latency is not None else None,
            "truth_team": truth_team,
            "predicted_team": predicted_team,
            "team_correct": (truth_team == predicted_team) if (
                truth_team in ("H", "A") and predicted_team in ("H", "A")) else None,
            "hint_team_correct": (truth_team == ("H" if team_hint == 1 else "A")) if (
                truth_team in ("H", "A") and team_hint in (1, -1)) else None,
            "model_team": model_team,
            "model_team_correct": (truth_team == model_team) if (
                truth_team in ("H", "A") and model_team in ("H", "A")) else None,
            "truth_match_count": len(truth_matches),
        })
    return rows


def print_summary(rows: list[dict]) -> None:
    print("restart audit: {} targeted set pieces".format(len(rows)))
    latency = [r["restart_latency_frames"] for r in rows if r["restart_latency_frames"] is not None]
    team = [r["team_correct"] for r in rows if r["team_correct"] is not None]
    hints = [r["hint_team_correct"] for r in rows if r["hint_team_correct"] is not None]
    model = [r["model_team_correct"] for r in rows if r["model_team_correct"] is not None]
    if latency:
        print("restart latency median {:+.1f} frames; mean {:+.1f}; abs mean {:.1f}".format(
            float(np.median(latency)),
            float(np.mean(latency)),
            float(np.mean(np.abs(np.asarray(latency, dtype=float))))))
    print("initial team accuracy with truth restart: {}/{}".format(sum(team), len(team)))
    print("named-tag team accuracy: {}/{}".format(sum(hints), len(hints)))
    if model:
        print("restart-team evidence accuracy: {}/{}".format(sum(model), len(model)))
    for row in rows:
        print(
            "{type:<22} tag {tag_range} truth {truth_dead_range} pred {predicted_dead_range} "
            "latency {restart_latency_frames!s} frames team {truth_team!s}->{predicted_team!s} "
            "hint={team_hint!s}".format(**row)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="YT.csv")
    parser.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out", help="optional CSV audit output")
    parser.add_argument("--no-public-gt", action="store_true")
    args = parser.parse_args()

    rows = audit(args.csv, None if args.no_public_gt else args.gt)
    print_summary(rows)

    if args.out:
        fields = list(rows[0].keys()) if rows else [
            "type", "team_hint", "tag_range", "truth_dead_range",
            "predicted_dead_range", "truth_restart", "predicted_restart",
            "restart_latency_frames", "restart_latency_seconds", "truth_team",
            "predicted_team", "team_correct", "hint_team_correct"]
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
