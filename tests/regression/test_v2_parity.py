"""Regression against v2: the v3 ``no_program_tokens`` variant is the v2 ProgramNet path.

Both are trained on the same synthetic fixture, the same patient folds and the same
feature space (frozen tiny encoder, PCA, morphometrics). They are not bit-identical
(v3 uses the heteroscedastic Gaussian head, a warm-up and cosine schedule, and the
composition plus smoothness terms through the v3 loop), so the check is that the
per-patient median Pearson on the planted neighbourhood program agrees within a
tolerance that is wide enough for a few hundred cells per patient and narrow enough to
catch a broken head mapping, which would send one of the two to chance.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from fixtures import small_ctx  # noqa: E402

TOL = 0.15


def _per_patient_median(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    rs = []
    for p in np.unique(groups):
        m = groups == p
        if m.sum() > 30 and np.isfinite(pred[m]).all() and pred[m].std() > 0:
            rs.append(np.corrcoef(pred[m], y[m])[0, 1])
    return float(np.median(rs)) if rs else float("nan")


@pytest.mark.slow
def test_no_program_tokens_matches_v2_programnet(tmp_path):
    pytest.importorskip("torch")
    from mmprog.components.ingest import Ingest
    from mmprog.components.preprocess import Preprocess
    from mmprog.components.program_inference import ProgramInference
    from mmprog.components.feature_extraction import FeatureExtraction
    from mmprog.components.feature_selection import FeatureSelection
    from mmprog.components.unified_model import UnifiedModel
    from mmprog.training.cv import make_folds, run_variant
    from mmprog.evaluation.cross_validation import build_cv_data, run_cv

    ctx = small_ctx(tmp_path, n_patients=4, cells_per=1500, max_epochs=12, batch_cells=256, warmup_steps=5,
                    patience=4, programs_max_iter=40, grid=4)
    for comp in (Ingest(), Preprocess(), ProgramInference(), FeatureExtraction(), FeatureSelection(), UnifiedModel()):
        ctx = comp.run(ctx)
    t = "prog_wnt_camdr_nb_z"
    j = ctx.targets.index(t)
    folds = make_folds(ctx.GROUPS, ctx.cfg.n_folds)

    # v3 path: the no_program_tokens variant through the v3 loop
    preds_v3, _, _, _, _, _ = run_variant("no_program_tokens", ctx, ctx.cfg, seed=7, folds=folds)
    r_v3 = _per_patient_median(preds_v3[:, j], ctx.Y[:, j], ctx.GROUPS)

    # v2 path: ProgramNet via the v2 cross-validation on the same features and targets. The v2 ``full``
    # model is f(features) + g(fourier(x, y), slide embedding); the v3 MLP path has no coordinate prior,
    # so the strict parity reference is v2 ``no_spatial`` (the same two-layer MLP and composition head).
    obs = ctx.obs.copy()
    obs["lineage"] = ctx.lineage
    data = build_cv_data(obs, list(ctx.targets), list(ctx.lin_names), ctx.cfg)
    preds_v2, y_used, _, _ = run_cv("no_spatial", data, ctx.EMB, ctx.MOR, ctx.cfg, ctx.device, seed=7)
    r_v2 = _per_patient_median(preds_v2[:, j], y_used[:, j], ctx.GROUPS)

    print(f"parity on {t}: v2 ProgramNet (no_spatial) r = {r_v2:.3f}, v3 no_program_tokens r = {r_v3:.3f}")
    assert np.isfinite(r_v2) and np.isfinite(r_v3)
    assert r_v2 > 0.2, "v2 reference path did not recover the planted program; the fixture is too small"
    assert abs(r_v2 - r_v3) < TOL, (r_v2, r_v3)

    # the unified model with neighbour tokens must not fall behind the v2 model with its coordinate prior
    preds_v2_full, y_used_full, _, _ = run_cv("full", data, ctx.EMB, ctx.MOR, ctx.cfg, ctx.device, seed=7)
    r_v2_full = _per_patient_median(preds_v2_full[:, j], y_used_full[:, j], ctx.GROUPS)
    preds_v3_full, _, _, _, _, _ = run_variant("full", ctx, ctx.cfg, seed=7, folds=folds)
    r_v3_full = _per_patient_median(preds_v3_full[:, j], ctx.Y[:, j], ctx.GROUPS)
    print(f"unified on {t}: v2 full (coordinate prior) r = {r_v2_full:.3f}, v3 full (neighbour tokens) r = {r_v3_full:.3f}")
    assert r_v3_full > r_v2_full - TOL, (r_v2_full, r_v3_full)
