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

from yt import control, dead, evalkit, features, machine
from yt.load import ROOT, load_match

warnings.simplefilter("ignore", RuntimeWarning)


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    control: control.ControlConfig = control.ControlConfig()
    dead: dead.DeadConfig = dead.DeadConfig()
    min_len: int = 25
    use_restart_team: bool = True


def build(cfg: PipelineConfig = PipelineConfig(), m=None, f=None) -> tuple[np.ndarray, dict]:
    """Labels for the whole match plus intermediate results for inspection."""
    m = m if m is not None else load_match()
    f = f if f is not None else features.compute(m)
    kickoffs = {}
    for h, (lo, hi) in enumerate(zip(*features.runs(m.graded)), start=1):
        kickoffs[int(lo)] = dead.kickoff_frame(m, f, int(lo), int(hi), h)
    d = dead.detect(f, m.graded, cfg.dead, kickoffs=kickoffs)
    c = control.detect(m, f, cfg.control)
    rt = dead.restart_teams(f, d.segments) if cfg.use_restart_team else {}
    labels = machine.label(c.ctrl, d.dead, m.graded, m.half, restart_team=rt,
                           min_len=cfg.min_len)
    return labels, {"match": m, "features": f, "dead": d, "control": c, "restart_team": rt}


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
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.variant:
        cfg = dataclasses.replace(
            cfg, control=dataclasses.replace(cfg.control, variant=args.variant))
    if args.min_len is not None:
        cfg = dataclasses.replace(cfg, min_len=args.min_len)
    if args.no_restart_team:
        cfg = dataclasses.replace(cfg, use_restart_team=False)
    labels, _ = build(cfg)
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
        print("  spellF1 bootstrap 95% [{:.4f}, {:.4f}]".format(
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
