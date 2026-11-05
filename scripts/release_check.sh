#!/usr/bin/env bash
# Release check: lint, the fast tests, one synthetic end-to-end run, and the release gates it produced.
#
# Exits non-zero on the first failure. The slow end-to-end tests (tests/e2e, three main.py
# subprocess runs; about 15 minutes on two CPU cores) are not part of this script; run
# `python -m pytest tests/e2e -q -p no:warnings` before tagging a release.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${MMPROG_CHECK_ROOT:-./workspace/release_check}"
RUN_NAME="release_check_synthetic"

echo "== 1/4 lint"
python -m ruff check src tests main.py --select E,F,W --ignore E501

echo "== 2/4 unit and regression tests"
python -m pytest tests/unit tests/regression -q -p no:warnings

echo "== 3/4 synthetic end-to-end run (CPU, no download)"
python main.py --config configs/synthetic.yaml --set run.root="$ROOT" --set run.run_name="$RUN_NAME"

echo "== 4/4 release gates"
RESULTS="$ROOT/runs/$RUN_NAME/RESULTS.json"
test -f "$RESULTS" || { echo "RESULTS.json not written: $RESULTS"; exit 1; }
python - "$RESULTS" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps(r["summary"], indent=2))
for name, g in r["gates"].items():
    print(f"  {name:<22} {g['status']:<14} value={g['value']}  threshold={g['threshold']}")
print("on the synthetic fixture the gates test the plumbing, not the biology" if r.get("demo_mode") else "")
EOF
echo "release check finished; full results in $RESULTS"
