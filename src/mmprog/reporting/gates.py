"""Release gates: the pass / fail checks a run must clear before its findings are reported as such.

Every gate is read from an artefact that an earlier component wrote to disk; nothing is
recomputed here, so a gate can always be traced to one file and one column. A gate whose
artefact or column is missing is reported as ``not_evaluated``, never as passed.

The gates and their sources:

    above_chance          evaluation/cv_results.csv        full - shuffled on the main target > 0.05
    beyond_identity       evaluation/cv_results.csv        full - composition_only on the main target > 0.05
    ci_excludes_zero      evaluation/cv_results.csv        lower 95 % bootstrap bound of full > 0
    shuffled_at_chance    evaluation/cv_results.csv        |shuffled| < 0.05
    calibrated            evaluation/calibration.csv       pooled coverage of the 90 % Gaussian interval in [0.85, 0.95]
    backbone_stable       interpretation/gene_backbone.csv at least 3 genes with selection_fraction >= 0.8 for the main program
    inference_parity      deployment/sanity_check.csv      parity_pearson (morphology-only path vs cross-validated prediction) > 0.95
    stage_association     evaluation/stage_association.csv PERMANOVA p < 0.05 or any BH-adjusted trend q < 0.1
    external_validation   external_validation/validation_summary.json  at least one validator ran, none failed or broke
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..logging_utils import say, show_table

NOT_EVALUATED = "not_evaluated"


@dataclass
class Gate:
    name: str
    passed: Any                 # True, False or "not_evaluated"
    value: Any                  # a float, or a short text (the external gate lists each validator's outcome)
    threshold: str
    source: str
    meaning: str

    @property
    def status(self) -> str:
        if self.passed is True:
            return "PASS"
        if self.passed is False:
            return "FAIL"
        return "NOT EVALUATED"


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _finite(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _fmt_value(v: Any) -> str:
    """A float to three decimals, text as is, nothing for None or NaN."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if not np.isfinite(f) else f"{f:.3f}"


def _ci_lower(text) -> Optional[float]:
    """Parse the lower bound of a '(lo, hi)' string as written in cv_results.csv."""
    if not isinstance(text, str):
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[0]) if nums else None


def _program_of(target: str) -> str:
    """'prog_wnt_camdr_nb_z' -> 'wnt_camdr'; 'anchor_wnt_camdr_cell_z' -> 'wnt_camdr'."""
    name = re.sub(r"^(prog|anchor)_", "", target)
    return re.sub(r"_(cell|nb)_z$", "", name)


# ---------------------------------------------------------------------- individual gates
def _cv_gates(run_dir: str, main_target: str) -> List[Gate]:
    src = os.path.join("evaluation", "cv_results.csv")
    res = _read_csv(os.path.join(run_dir, src))
    row = None
    if res is not None and "target" in res.columns and (res["target"] == main_target).any():
        row = res.set_index("target").loc[main_target]

    def pick(col: str) -> Optional[float]:
        return _finite(row[col]) if row is not None and col in row.index else None

    full, shuffled, comp = pick("full"), pick("shuffled"), pick("composition_only")
    lower = _ci_lower(row["full_95ci"]) if row is not None and "full_95ci" in row.index else None
    gates = []
    v = None if full is None or shuffled is None else full - shuffled
    gates.append(Gate("above_chance", NOT_EVALUATED if v is None else bool(v > 0.05), v, "> 0.05", src,
                      "the full model beats the shuffled-label baseline on the main target by a margin"))
    v = None if full is None or comp is None else full - comp
    gates.append(Gate("beyond_identity", NOT_EVALUATED if v is None else bool(v > 0.05), v, "> 0.05", src,
                      "morphology carries program information beyond the cell-type composition baseline"))
    gates.append(Gate("ci_excludes_zero", NOT_EVALUATED if lower is None else bool(lower > 0), lower, "> 0", src,
                      "the lower 95 % patient-bootstrap bound of the full model's correlation is above zero"))
    gates.append(Gate("shuffled_at_chance", NOT_EVALUATED if shuffled is None else bool(abs(shuffled) < 0.05), shuffled,
                      "|r| < 0.05", src, "the shuffled-label baseline sits at chance, so no leak feeds the model"))
    return gates


def _calibration_gate(run_dir: str, main_target: str) -> Gate:
    src = os.path.join("evaluation", "calibration.csv")
    tab = _read_csv(os.path.join(run_dir, src))
    v = None
    if tab is not None and {"target", "level", "coverage_pooled"}.issubset(tab.columns):
        sel = tab[(tab["target"] == main_target) & (np.isclose(tab["level"].astype(float), 0.9))]
        if "kind" in tab.columns:
            sel = sel[sel["kind"] == "gaussian"] if (sel["kind"] == "gaussian").any() else sel
        if len(sel):
            v = _finite(sel["coverage_pooled"].iloc[0])
    return Gate("calibrated", NOT_EVALUATED if v is None else bool(0.85 <= v <= 0.95), v, "in [0.85, 0.95]", src,
                "the 90 % Gaussian interval covers about 90 % of held-out cells (pooled; read the per-patient spread too)")


def _backbone_gate(run_dir: str, main_target: str) -> Gate:
    src = os.path.join("interpretation", "gene_backbone.csv")
    tab = _read_csv(os.path.join(run_dir, src))
    v = None
    if tab is not None and {"program", "gene", "selection_fraction"}.issubset(tab.columns):
        prog = _program_of(main_target)
        sel = tab[(tab["program"].astype(str) == prog) & (tab["selection_fraction"].astype(float) >= 0.8)]
        v = float(sel["gene"].nunique())
    return Gate("backbone_stable", NOT_EVALUATED if v is None else bool(v >= 3), v, ">= 3 genes", src,
                "the main program has at least three genes that survive stability selection over half-patient refits")


def _parity_gate(run_dir: str) -> Gate:
    """sanity_check.csv: parity_pearson (morphology-only path vs training path on the same cells) and, when the
    inference path raised, parity_error with the exception text. An error is a FAIL, not a missing number: the
    drift the gate exists to catch is exactly a crash in the deployed path."""
    src = os.path.join("deployment", "sanity_check.csv")
    tab = _read_csv(os.path.join(run_dir, src))
    v = None
    error = ""
    if tab is not None and len(tab):
        if "parity_error" in tab.columns:
            raw = tab["parity_error"].iloc[0]
            error = "" if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else str(raw).strip()
        if "parity_pearson" in tab.columns:
            v = _finite(tab["parity_pearson"].iloc[0])
    if error:
        passed: Any = False
    elif v is None:
        passed = NOT_EVALUATED
    else:
        passed = bool(v > 0.95)
    meaning = "the morphology-only inference path reproduces the training-path predictions on a training slide"
    if error:
        meaning += f" (inference path raised: {error[:120]})"
    return Gate("inference_parity", passed, v, "> 0.95 and no inference error", src, meaning)


def _stage_gate(run_dir: str) -> Gate:
    """stage_association.csv columns: vectors, test, program, statistic, p_value, q_value, ..., note.

    Only the rows on the predicted vectors count; a row whose note starts with "skipped:" is not evaluated.
    """
    src = os.path.join("evaluation", "stage_association.csv")
    tab = _read_csv(os.path.join(run_dir, src))
    v = None
    passed: Any = NOT_EVALUATED
    if tab is not None and len(tab) and {"test", "p_value"}.issubset(tab.columns):
        if "vectors" in tab.columns:
            tab = tab[tab["vectors"].astype(str) == "predicted"]
        if "note" in tab.columns:
            tab = tab[~tab["note"].astype(str).str.startswith("skipped")]
        test = tab["test"].astype(str).str.lower()
        perm = tab[test == "permanova"]
        p_perm = _finite(perm["p_value"].iloc[0]) if len(perm) else None
        q_trend = None
        if "q_value" in tab.columns:
            trend = tab[test == "jt_trend"]
            if len(trend):
                q_trend = _finite(np.nanmin(pd.to_numeric(trend["q_value"], errors="coerce").to_numpy(dtype=float)))
        if p_perm is not None or q_trend is not None:
            perm_ok = p_perm is not None and p_perm < 0.05
            trend_ok = q_trend is not None and q_trend < 0.1
            passed = bool(perm_ok or trend_ok)
            # report the number that decided the gate: the PERMANOVA p when it passed, else the smallest trend q
            v = p_perm if (perm_ok or q_trend is None) else q_trend
    return Gate("stage_association", passed, v, "PERMANOVA p < 0.05 or trend q < 0.1", src,
                "the patient-level vector of predicted program burdens is associated with disease stage")


def _external_validation_gate(run_dir: str) -> Gate:
    """validation_summary.json: validator -> {status: ran|skipped|failed, passed: bool (set by the validator from its
    own stated bar), reason}.

    The gate passes only when at least one validator ran, every validator that ran reports ``passed`` True and none
    failed. A validator that ran without writing ``passed`` makes the gate NOT EVALUATED: a number nobody judged is
    not a pass. The gate value string names each validator with its outcome so the reason is visible in the table.
    """
    src = os.path.join("external_validation", "validation_summary.json")
    path = os.path.join(run_dir, src)
    passed: Any = NOT_EVALUATED
    v: Any = None
    if os.path.exists(path):
        try:
            with open(path) as fh:
                summary = json.load(fh)
        except Exception:
            summary = {}
        entries = {k: d for k, d in summary.items() if isinstance(d, dict) and "status" in d}
        ran = {k: d for k, d in entries.items() if str(d.get("status", "")).lower() == "ran"}
        broken = [k for k, d in entries.items() if str(d.get("status", "")).lower() == "failed"]
        unjudged = [k for k, d in ran.items() if d.get("passed") not in (True, False)]
        not_passed = [k for k, d in ran.items() if d.get("passed") is False]
        parts = []
        for k, d in entries.items():
            st = str(d.get("status", "")).lower()
            if st == "ran":
                parts.append(f"{k}:{'pass' if d.get('passed') is True else ('fail' if d.get('passed') is False else 'unjudged')}")
            else:
                parts.append(f"{k}:{st}")
        v = "; ".join(parts) if parts else None
        if broken or not_passed:
            passed = False
        elif ran and not unjudged:
            passed = True
        else:
            passed = NOT_EVALUATED
    return Gate("external_validation", passed, v, "validators ran >= 1, every ran validator passed, failed = 0", src,
                "every validator that could run on the available data met its own stated bar; skipped validators are listed with their reason")


# ---------------------------------------------------------------------- public API
def evaluate_gates(run_dir: str, main_target: str) -> List[Gate]:
    gates = _cv_gates(run_dir, main_target)
    gates.append(_calibration_gate(run_dir, main_target))
    gates.append(_backbone_gate(run_dir, main_target))
    gates.append(_parity_gate(run_dir))
    gates.append(_stage_gate(run_dir))
    gates.append(_external_validation_gate(run_dir))
    return gates


def gates_table(gates: List[Gate]) -> pd.DataFrame:
    rows = []
    for g in gates:
        d = asdict(g)
        d["status"] = g.status
        rows.append(d)
    return pd.DataFrame(rows, columns=["name", "status", "passed", "value", "threshold", "source", "meaning"])


def gates_dict(gates: List[Gate]) -> Dict[str, dict]:
    return {g.name: dict(passed=g.passed, status=g.status, value=g.value, threshold=g.threshold, source=g.source,
                         meaning=g.meaning) for g in gates}


def write_gates(run_dir: str, main_target: str, gates: List[Gate], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write reporting/release_gates.csv and <run_dir>/RESULTS.json; return the RESULTS dict."""
    os.makedirs(os.path.join(run_dir, "reporting"), exist_ok=True)
    tab = gates_table(gates)
    tab.to_csv(os.path.join(run_dir, "reporting", "release_gates.csv"), index=False)
    n_pass = int(sum(g.passed is True for g in gates))
    n_fail = int(sum(g.passed is False for g in gates))
    n_na = len(gates) - n_pass - n_fail
    results = dict(main_target=main_target, gates=gates_dict(gates),
                   summary=dict(passed=n_pass, failed=n_fail, not_evaluated=n_na, total=len(gates)))
    if extra:
        results.update(extra)
    with open(os.path.join(run_dir, "RESULTS.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    say(f"release gates: {n_pass} passed, {n_fail} failed, {n_na} not evaluated; written to RESULTS.json and reporting/release_gates.csv")
    return results


def print_gates(gates: List[Gate], demo_mode: bool = False) -> None:
    tab = gates_table(gates)[["name", "status", "value", "threshold", "meaning"]].copy()
    tab["value"] = tab["value"].map(_fmt_value)
    show_table(tab, title="release gates (each read from one artefact; NOT EVALUATED means the artefact or column is absent):")
    n_pass = sum(g.passed is True for g in gates)
    n_fail = sum(g.passed is False for g in gates)
    say(f"gates: {n_pass} of {len(gates)} passed, {n_fail} failed, {len(gates) - n_pass - n_fail} not evaluated")
    if demo_mode:
        say("this is the synthetic fixture: the gates test the plumbing, not the biology")


def gates_markdown(gates: List[Gate]) -> List[str]:
    lines = ["| gate | status | value | threshold | source | meaning |", "|---|---|---|---|---|---|"]
    for g in gates:
        val = _fmt_value(g.value)
        lines.append(f"| {g.name} | {g.status} | {val} | {g.threshold} | `{g.source}` | {g.meaning} |")
    return lines
