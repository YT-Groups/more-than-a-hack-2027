"""Sanity checks for candidate ball-possession label files."""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames


FPS = GAMES[GAME]["fps"]


def find_dead_runs(labels: np.ndarray, lo: int = 0, hi: int | None = None
                   ) -> list[tuple[int, int]]:
    """Find half-open runs where labels are D within [lo, hi)."""
    labels = np.asarray(labels)
    if hi is None:
        hi = len(labels)
    mask = labels[lo:hi] == "D"
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(lo + int(a), lo + int(b)) for a, b in zip(starts, ends)]


def dead_runs_per_half(labels: np.ndarray, blocks=None) -> dict[int, list[tuple[int, int]]]:
    """Find D runs separately in each graded half block."""
    labels = np.asarray(labels)
    if blocks is None:
        blocks = half_video_blocks(GAME)
    return {
        int(half): find_dead_runs(labels, first, last + 1)
        for half, first, last in blocks
    }


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Half-open interval overlap; touching endpoints does not overlap."""
    return a0 < b1 and a1 > b0


def restart_timings(runs: list[tuple[int, int, int]], tags: list[dict],
                    n_frames: int) -> dict[str, dict]:
    """Split tag matches into one, ambiguous, and no-match timing groups."""
    timings = {}
    for tag in tags:
        a = max(0, min(n_frames, tag["start"]))
        b = max(0, min(n_frames, tag["end"]))
        ends = sorted(run_end for _, run_end, _ in runs if a <= run_end <= b)
        tag_type = tag.get("type") or "N/A"
        result = timings.setdefault(tag_type, {"one": [], "many": [], "none": 0})
        if not ends:
            result["none"] += 1
        elif len(ends) == 1:
            result["one"].append((ends[0] - a) / FPS)
        else:
            result["many"].append([(end - a) / FPS for end in ends])
    return timings


def load_tags(xml_path: str | None = None) -> list[dict]:
    if xml_path is None:
        xml_path = asset_paths(game=GAME, source="tactical").get("smart_tagging")
        if xml_path is None:
            raise FileNotFoundError("tactical source has no Smart Tagging asset")
    tags = []
    for instance in ET.parse(xml_path).iter("instance"):
        code = instance.findtext("code") or ""
        if code.strip().upper() != "SET PIECES":
            continue
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        a = int(round(start_s * FPS))
        b = int(round(end_s * FPS))
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        tags.append({"start": a, "end": b, "start_s": start_s, "end_s": end_s,
                     "raw": instance.findtext("code"), "code": code.strip().upper(),
                     "type": labels.get("Type"), "team": labels.get("Team"),
                     "half": labels.get("Half")})
    return tags


def read_submission(path: str) -> tuple[np.ndarray, list[int]]:
    labels = []
    frames = []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        names = [name.strip() for name in reader.fieldnames]
        if len(names) != 2 or names[0].lower() != "frame" or names[1].lower() != "label":
            raise ValueError("header must be frame,label")
        for row in reader:
            frames.append(int(row[reader.fieldnames[0]]))
            label = str(row[reader.fieldnames[1]]).strip().upper()
            if label not in ("H", "A", "D", "X"):
                raise ValueError("invalid label {!r}".format(label))
            labels.append(label)
    return np.asarray(labels, dtype="<U1"), frames


def public_truth_dead_latency(labels: np.ndarray,
                              gt_path: str | None = None) -> list[dict]:
    """Compare predicted D-to-live ends with public ground-truth D-run ends."""
    if gt_path is None:
        gt_path = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    truth_labels = []
    with open(gt_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            truth_labels.append(str(row["label"]).strip().upper())
    truth = np.asarray(truth_labels, dtype="<U1")

    lo, hi = 11185, 18685
    truth_runs = find_dead_runs(truth, lo, hi)
    predicted_runs = find_dead_runs(labels, lo, hi)
    result = []
    for truth_start, truth_end in truth_runs:
        if predicted_runs:
            predicted_end = min(
                (run_end for _, run_end in predicted_runs),
                key=lambda run_end: abs(run_end - truth_end),
            )
            predicted_start = min(
                (run_start for run_start, _ in predicted_runs),
                key=lambda run_start: abs(run_start - truth_start),
            )
            latency = predicted_end - truth_end
            start_latency = predicted_start - truth_start
        else:
            predicted_end = None
            latency = None
            predicted_start = None
            start_latency = None
        result.append({
            "truth_range": [truth_start, truth_end],
            "predicted_end": predicted_end,
            "latency_frames": latency,
            "latency_seconds": None if latency is None else latency / FPS,
            "predicted_start": predicted_start,
            "start_latency_frames": start_latency,
            "start_latency_seconds": None if start_latency is None else start_latency / FPS,
        })
    return result


def summarize(labels: np.ndarray, blocks=None, tags=None) -> dict:
    labels = np.asarray(labels)
    n = source_frames("tactical", GAME)
    if len(labels) != n:
        raise ValueError("expected {} labels, got {}".format(n, len(labels)))
    if blocks is None:
        blocks = half_video_blocks(GAME)

    half_runs = dead_runs_per_half(labels, blocks)
    runs = [(a, b, half) for half, items in half_runs.items() for a, b in items]
    durations = np.asarray([(b - a) / FPS for a, b, _ in runs], dtype=float)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    dead_share = float(np.mean(labels[graded] == "D")) if graded.any() else 0.0

    clipped_tags = []
    for tag in tags or []:
        a = max(0, min(n, tag["start"]))
        b = max(0, min(n, tag["end"]))
        if b > a:
            clipped_tags.append((a, b))

    untagged = [item for item in runs
                if (item[1] - item[0]) / FPS > 3.0
                and not any(overlaps(item[0], item[1], t0, t1) for t0, t1 in clipped_tags)]
    tagged_dead = [item for item in runs
                   if any(overlaps(item[0], item[1], t0, t1) for t0, t1 in clipped_tags)]
    tags_containing_dead = [
        (t0, t1, [item for item in runs if overlaps(t0, t1, item[0], item[1])])
        for t0, t1 in clipped_tags
    ]
    tags_by_type = {}
    for tag in tags or []:
        a = max(0, min(n, tag["start"]))
        b = max(0, min(n, tag["end"]))
        if b <= a:
            continue
        tag_type = tag.get("type") or "N/A"
        matches = [item for item in runs if overlaps(a, b, item[0], item[1])]
        result = tags_by_type.setdefault(tag_type, {"total": 0, "covered": 0})
        result["total"] += 1
        if matches:
            result["covered"] += 1
    longest = sorted(runs, key=lambda x: x[1] - x[0], reverse=True)[:5]

    percentiles = {}
    if durations.size:
        percentiles = {
            "p10_s": float(np.percentile(durations, 10)),
            "p50_s": float(np.percentile(durations, 50)),
            "p90_s": float(np.percentile(durations, 90)),
            "max_s": float(durations.max()),
        }
    else:
        percentiles = {"p10_s": 0.0, "p50_s": 0.0, "p90_s": 0.0, "max_s": 0.0}
    return {
        "dead_share": dead_share,
        "runs_per_half": {h: len(items) for h, items in half_runs.items()},
        "n_runs": len(runs),
        "durations_s": durations,
        "under_2s": [item for item in runs if (item[1] - item[0]) / FPS < 2.0],
        "over_60s": [item for item in runs if (item[1] - item[0]) / FPS > 60.0],
        "untagged_over_3s": untagged,
        "tagged_dead": tagged_dead,
        "tags_containing_dead": tags_containing_dead,
        "tags_by_type": tags_by_type,
        "longest": longest,
        "restart_timing": restart_timings(runs, tags or [], n),
        "public_truth_dead_latency": public_truth_dead_latency(labels),
        **percentiles,
    }


def print_report(path: str, xml_path: str | None = None) -> dict:
    labels, frames = read_submission(path)
    if frames != list(range(len(frames))):
        bad = next((i for i, frame in enumerate(frames) if frame != i), None)
        raise ValueError("frame column mismatch at row/index {}".format(bad))

    tags = load_tags(xml_path) if xml_path is not None else None
    if tags is None:
        try:
            tags = load_tags()
        except (FileNotFoundError, OSError, ET.ParseError):
            tags = []
    report = summarize(labels, tags=tags)

    print("candidate {}".format(path))
    print("graded dead share {:.4f}".format(report["dead_share"]))
    print("dead stretches per half {}".format(report["runs_per_half"]))
    print("total {}  10/50/90/max s: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
        report["n_runs"], report["p10_s"], report["p50_s"], report["p90_s"], report["max_s"]))
    print("under 2 s: {}  over 60 s: {}".format(
        len(report["under_2s"]), len(report["over_60s"])))
    print("Smart Tagging: {} windows; {} dead stretches overlap tags; {} untagged stretches over 3 s".format(
        len(report["tags_containing_dead"]),
        len(report["tagged_dead"]),
        len(report["untagged_over_3s"])))
    covered = sum(1 for _, _, dead_runs in report["tags_containing_dead"] if dead_runs)
    print("set-piece windows containing a dead stretch: {}/{}".format(
        covered, len(report["tags_containing_dead"])))
    for tag_type, counts in sorted(report["tags_by_type"].items()):
        print("  {}: {}/{}".format(
            tag_type, counts["covered"], counts["total"]))
    print("restart timing by set-piece type (dead end - tag start)")
    for tag_type, timing in sorted(report["restart_timing"].items()):
        values = timing["one"]
        if values:
            array = np.asarray(values)
            print("  {}: n={} median={:.2f}s p10={:.2f}s p90={:.2f}s ambiguous={} none={}".format(
                tag_type, len(array), float(np.median(array)),
                float(np.percentile(array, 10)), float(np.percentile(array, 90)),
                len(timing["many"]), timing["none"]))
        else:
            print("  {}: n=0 ambiguous={} none={}".format(
                tag_type, len(timing["many"]), timing["none"]))
    truth_latency = report["public_truth_dead_latency"]
    print("public truth dead-edge latency (detector - truth)")
    for item in truth_latency:
        start, end = item["truth_range"]
        print("  truth [{}, {}) detector=[{}, {}) start_latency={} frames ({:+.2f}s) end_latency={} frames ({:+.2f}s)".format(
            start, end, item["predicted_start"], item["predicted_end"],
            item["start_latency_frames"], item["start_latency_seconds"],
            item["latency_frames"], item["latency_seconds"]))

    print("longest five")
    for a, b, half in report["longest"]:
        print("  H{} [{}, {})  {:.2f} s".format(half, a, b, (b - a) / FPS))
    print("under 2 s ranges")
    for a, b, half in report["under_2s"]:
        print("  H{} [{}, {})".format(half, a, b))
    print("untagged stretches over 3 s")
    for a, b, half in report["untagged_over_3s"]:
        print("  H{} [{}, {})  {:.2f} s".format(half, a, b, (b - a) / FPS))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="candidate frame,label CSV")
    parser.add_argument("--xml", help="optional Smart Tagging XML")
    args = parser.parse_args()
    print_report(args.csv, args.xml)


if __name__ == "__main__":
    main()
