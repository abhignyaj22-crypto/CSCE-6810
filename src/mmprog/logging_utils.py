"""Printing and run logging.

Every stage prints what it is doing, what it found and what it decided. The same
text is appended to ``run.log`` in the run directory, so a Colab session that
disconnects still leaves a readable trace. Plain ``print`` is used deliberately:
the pipeline is read by people, not log aggregators.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import os
import sys
import time
from typing import Iterator, Optional

import numpy as np
import pandas as pd

_LOG_PATH: Optional[str] = None


def attach_log_file(path: str) -> None:
    """Tee every later ``say`` call to this file."""
    global _LOG_PATH
    _LOG_PATH = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(f"\n===== run started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")


def say(*parts: object, indent: int = 0) -> None:
    """Print a line with a timestamp and mirror it to the run log."""
    text = " ".join(str(p) for p in parts)
    line = f"[{_dt.datetime.now():%H:%M:%S}] {'  ' * indent}{text}"
    print(line, flush=True)
    if _LOG_PATH:
        with open(_LOG_PATH, "a") as fh:
            fh.write(line + "\n")


def banner(title: str) -> None:
    rule = "=" * max(60, len(title) + 8)
    say(rule)
    say(f"    {title}")
    say(rule)


def section(title: str) -> None:
    say("-" * 60)
    say(title)
    say("-" * 60)


def show_table(df: pd.DataFrame, max_rows: int = 40, title: Optional[str] = None) -> None:
    """Print a DataFrame in full width, head-limited, mirrored to the log."""
    if title:
        say(title)
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 40):
        text = df.head(max_rows).to_string()
    for line in text.splitlines():
        say(line, indent=1)
    if len(df) > max_rows:
        say(f"... {len(df) - max_rows} more rows", indent=1)


def describe_array(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if a.size == 0:
        say(f"{name}: empty array shape={a.shape}")
        return
    sample = a if a.size <= 1_000_000 else a.ravel()[:: max(1, a.size // 1_000_000)]
    say(f"{name}: shape={a.shape} dtype={a.dtype} min={np.nanmin(sample):.4g} "
        f"median={np.nanmedian(sample):.4g} max={np.nanmax(sample):.4g} "
        f"nan={int(np.isnan(sample).sum()) if np.issubdtype(a.dtype, np.floating) else 0}")


@contextlib.contextmanager
def timed(label: str) -> Iterator[None]:
    """``with timed("embedding"):`` prints the elapsed time when the block ends."""
    t0 = time.time()
    say(f"{label} ...")
    try:
        yield
    finally:
        say(f"{label} done in {time.time() - t0:.1f}s")


def progress(i: int, n: int, every: int = 10, label: str = "") -> None:
    """Print a progress line every ``every`` percent."""
    if n <= 0:
        return
    step = max(1, n * every // 100)
    if i % step == 0 or i == n - 1:
        say(f"{label} {i + 1}/{n} ({100 * (i + 1) / n:.0f}%)", indent=1)


def warn(*parts: object) -> None:
    say("WARNING:", *parts)
    print(" ".join(str(p) for p in parts), file=sys.stderr)
