#!/usr/bin/env python
"""Legacy v2 runner, kept for the regression tests; the entry point is main.py.

Runs the v2 stage pipeline (mmprog.pipeline).

Examples
    python scripts/run_pipeline.py --config configs/synthetic.yaml
    python scripts/run_pipeline.py --config configs/smoke.yaml --set cv_variants=[full,shuffled]
    python scripts/run_pipeline.py --config configs/colab_full.yaml --stop-after s3_features
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mmprog.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
