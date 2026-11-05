"""Legacy v2 runner, kept for the regression tests; the entry point is main.py.

``mmprog --config configs/synthetic.yaml --set sample_limit=4`` runs the v2 stage pipeline.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import load_config
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmprog", description="Morphology to drug-resistance program scores in multiple myeloma")
    p.add_argument("--config", "-c", default=None, help="YAML configuration file (configs/*.yaml)")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                   help="override a configuration key; repeatable (e.g. --set sample_limit=4 --set cv_variants=[full,shuffled])")
    p.add_argument("--stop-after", default=None, choices=Pipeline.STAGES, help="run the stages up to and including this one")
    p.add_argument("--list-stages", action="store_true", help="print the stage order and exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_stages:
        for s in Pipeline.STAGES:
            print(s)
        return 0
    cfg = load_config(args.config, args.set)
    Pipeline(cfg).run_all(stop_after=args.stop_after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
