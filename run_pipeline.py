"""One command: match data -> YT.csv -> validate -> public-window score -> proxy checks.

    python run_pipeline.py --out YT.csv

Deterministic: no randomness, no dependence on time or file order. Data is read through
games.asset_paths(), so MTAH_DATA_DIR moves it without code changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import warnings

import numpy as np

from yt import control, dead, evalkit, evidence, features, laws, machine, reception, tags, teamfix
from yt.load import ROOT, load_match

warnings.simplefilter("ignore", RuntimeWarning)


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    control: control.ControlConfig = control.ControlConfig()
    dead: dead.DeadConfig = dead.DeadConfig()
    min_len: int = 25
    use_restart_team: bool = True
    use_tags: bool = True
    # Laws of the Game refinements (yt/laws.py, yt/teamfix.py)
    use_kickoff_timing: bool = True    # Law 8: restarts at a kick-off pattern move onto the kick
    use_goal_merge: bool = True        # Laws 8+10: no live play between a goal and its kick-off
    use_teamfix: bool = True           # Law 8: tracklets in the wrong half at a kick-off
    use_receptions: bool = False
    reception: reception.ReceptionConfig = reception.ReceptionConfig()


def _dead_ball(cfg: "PipelineConfig", m, f, events: list):
    """Dead-ball result, half-start kick-offs, and the law-based refinements if enabled."""
    kickoffs = {}
    for h, (lo, hi) in enumerate(zip(*features.runs(m.graded)), start=1):
        kickoffs[int(lo)] = dead.kickoff_frame(m, f, int(lo), int(hi), h)
    prior = tags.prior_mask(events, m.n) if cfg.use_tags else None
    d = dead.detect(f, m.graded, cfg.dead, kickoffs=kickoffs, prior=prior)
    if cfg.use_kickoff_timing:
        d = laws.align_kickoffs(m, f, d)
    if cfg.use_goal_merge:
        d = laws.merge_before_kickoffs(d, m.graded)
    return d, kickoffs


def build(cfg: PipelineConfig = PipelineConfig(), m=None, f=None) -> tuple[np.ndarray, dict]:
    """Labels for the whole match plus intermediate results for inspection."""
    m = m if m is not None else load_match()
    f = f if f is not None else features.compute(m)
    events = tags.load() if cfg.use_tags else []
    d, kickoffs = _dead_ball(cfg, m, f, events)
    team_fixes = []
    if cfg.use_teamfix:
        # Law 8: at every kick-off each player is in his own half. Tracklets that are not carry
        # the wrong team label; move them to the other team and rebuild from the features up.
        ks = sorted({k for k in kickoffs.values() if k is not None} | set(laws.kickoff_frames(d)))
        team_fixes = teamfix.kickoff_violations(m, ks)
        if team_fixes:
            m = teamfix.fix_tracklets(m, team_fixes)
            f = features.compute(m)
            d, kickoffs = _dead_ball(cfg, m, f, events)
    c = control.detect(m, f, cfg.control)
    receptions = {}
    if cfg.use_receptions:
        receptions_result = reception.detect(m, f, c.ctrl, d.dead, cfg.reception)
        receptions = receptions_result.handover
    rt = dead.restart_teams(f, d.segments) if cfg.use_restart_team else {}
    if cfg.use_restart_team and cfg.use_tags:
        rt = tags.fill_restart_teams(rt, d.segments, tags.team_hints(events))
    dynamic_restarts = {}
    for onset, restart, _, _ in d.segments:
        if restart == onset:
            continue
        frame, confidence, detail = evidence.dynamic_restart_alignment(m, f, restart)
        dynamic_restarts[int(restart)] = {
            "frame": frame,
            "confidence": confidence,
            **detail,
        }
    labels = machine.label(c.ctrl, d.dead, m.graded, m.half, restart_team=rt,
                           min_len=cfg.min_len, receptions=receptions)
    return labels, {
        "match": m,
        "features": f,
        "dead": d,
        "control": c,
        "restart_team": rt,
        "team_fixes": team_fixes,
        "dynamic_restarts": dynamic_restarts,
        "receptions": receptions_result if cfg.use_receptions else reception.ReceptionResult({}, {}, []),
    }


def write_csv(path: str, labels: np.ndarray) -> None:
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, x) for i, x in enumerate(labels))


def settings_dict(cfg: PipelineConfig) -> dict:
    return dataclasses.asdict(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="YT.csv")
    ap.add_argument("--log", metavar="NAME", help="append a row to reports/ablation.csv")
    ap.add_argument("--note", default="")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--variant", choices=["C1", "C2", "C3"], help="control variant override")
    ap.add_argument("--min-len", type=int, help="minimum possession length override")
    ap.add_argument("--no-restart-team", action="store_true")
    ap.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    ap.add_argument("--no-laws", nargs="*", metavar="RULE",
                    choices=["kickoff_timing", "goal_merge", "teamfix"],
                    help="switch off law-based refinements (all if none named)")
    ap.add_argument("--receptions", action="store_true",
                    help="enable confirmed pass-reception ownership evidence")
    ap.add_argument("--reception-dwell", type=int)
    ap.add_argument("--reception-radius", type=float)
    ap.add_argument("--reception-margin", type=float)
    ap.add_argument("--no-reception-control", action="store_true")
    ap.add_argument("--audit-receptions", action="store_true")
    ap.add_argument("--audit-architecture", action="store_true",
                    help="print dynamic restart evidence and proxy architecture summary")
    ap.add_argument("--no-cache", action="store_true",
                    help="parse the raw ATD instead of reading data/cache")
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.variant:
        cfg = dataclasses.replace(
            cfg, control=dataclasses.replace(cfg.control, variant=args.variant))
    if args.min_len is not None:
        cfg = dataclasses.replace(cfg, min_len=args.min_len)
    if args.no_restart_team:
        cfg = dataclasses.replace(cfg, use_restart_team=False)
    if args.no_tags:
        cfg = dataclasses.replace(cfg, use_tags=False)
    if args.no_laws is not None:
        off = args.no_laws or ["kickoff_timing", "goal_merge", "teamfix"]
        cfg = dataclasses.replace(cfg, **{"use_" + r: False for r in off})
    if args.receptions:
        cfg = dataclasses.replace(cfg, use_receptions=True)
    rcfg = cfg.reception
    if args.reception_dwell is not None:
        rcfg = dataclasses.replace(rcfg, receiver_dwell=args.reception_dwell)
    if args.reception_radius is not None:
        rcfg = dataclasses.replace(rcfg, receiver_radius=args.reception_radius)
    if args.reception_margin is not None:
        rcfg = dataclasses.replace(rcfg, team_margin=args.reception_margin)
    if args.no_reception_control:
        rcfg = dataclasses.replace(rcfg, require_control=False)
    cfg = dataclasses.replace(cfg, reception=rcfg, use_receptions=(
        cfg.use_receptions or args.audit_receptions or args.reception_dwell is not None
        or args.reception_radius is not None or args.reception_margin is not None
        or args.no_reception_control))
    labels, result = build(cfg, m=load_match(use_cache=not args.no_cache))
    if args.audit_receptions:
        print("reception audit", json.dumps(result["receptions"].counts, sort_keys=True))
    if args.audit_architecture:
        dynamic = result["dynamic_restarts"]
        confidences = np.asarray([v["confidence"] for v in dynamic.values()], dtype=float)
        frames = np.asarray([
            v["frame"] - restart for restart, v in dynamic.items()
            if v["frame"] is not None
        ], dtype=float)
        print("architecture dynamic restart audit count {}".format(len(dynamic)))
        print("  median confidence {:.4f}".format(
            float(np.median(confidences)) if confidences.size else 0.0))
        print("  median edge displacement {} frames".format(
            int(np.median(frames)) if frames.size else 0))
    write_csv(args.out, labels)
    print("wrote", args.out)

    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                    args.out], check=True)

    m_graded = _graded()
    pub = evalkit.score_public(labels, bootstrap_n=args.bootstrap)
    prox = evalkit.proxies(labels, m_graded)
    print("public window  macroF1 {:.4f}  spellF1 {:.4f}  spells {} (gt {})".format(
        pub["macro_f1"], pub["spell_f1_mean"], pub["n_pred_spells"], pub["n_gt_spells"]))
    if args.bootstrap:
        print("  spellF1 bootstrap 90% (5th-95th pct) [{:.4f}, {:.4f}]".format(
            pub["spell_bootstrap_low"], pub["spell_bootstrap_high"]))
    print("full match proxies", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in prox.items()}))
    if args.log:
        evalkit.log_ablation(args.log, settings_dict(cfg), pub, prox, args.note)


def _graded() -> np.ndarray:
    from yt.load import graded_mask
    from games import source_frames
    return graded_mask(source_frames("tactical"))[0]


if __name__ == "__main__":
    main()
