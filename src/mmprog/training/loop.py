"""Training loop for the unified model on one GPU (or the CPU test path).

Loss per batch, on the real (non-padded) cells only:
    L = NLL(mean, log_var; y)                      heteroscedastic Gaussian head (or MSE when uncertainty = none)
      + lambda_comp   * cross_entropy(comp_logits, lineage)
      + lambda_smooth * mean over training neighbours of (mean_i - pred_nb)^2   (pred_nb from a detached cache)
      + lambda_align  * InfoNCE(morph_pooled, expression_embedding)   (TANGLE-style, adapted to cell level)

H100 settings (facts in ARCHITECTURE_V3.md section 8): bf16 autocast runs without a
GradScaler: bf16 keeps the float32 exponent range, so the underflow the scaler addresses
does not arise (author's reasoning; the PyTorch amp page states only the float16 case,
ARCHITECTURE_V3.md section 8). amp_dtype float16 is accepted by the config but refused
here for training, because no GradScaler is implemented and float16 would train unscaled
and silently underflow; use bfloat16 on Ampere, Ada or Hopper GPUs. TF32 is
enabled for matmul and cuDNN; ``torch.compile`` with ``dynamic=False`` is safe because every
batch has the same shape (the batcher pads the last one). On CPU the same code runs
without autocast.

Early stopping and temperature scaling use the inner validation patient only. The
temperature is a scalar multiplier on the predictive standard deviation chosen by a grid
search that minimises the validation NLL; it leaves the mean unchanged.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..logging_utils import say
from .data import CellBatcher, to_device


# ----------------------------------------------------------------------------- losses
def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian negative log-likelihood, 0.5 * (log sigma^2 + (y - mu)^2 / sigma^2)."""
    return 0.5 * (log_var + (y - mean) ** 2 * torch.exp(-log_var))


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE between two views of the same cells (rows are positives)."""
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _heads():
    """Prefer the shared head functions from models.heads when present; fall back to the local ones."""
    try:
        from ..models import heads as h
        return (getattr(h, "gaussian_nll", gaussian_nll), getattr(h, "info_nce", info_nce))
    except Exception:
        return gaussian_nll, info_nce


def batch_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Any, flags: Dict[str, bool]
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
    nll_fn, nce_fn = _heads()
    real = batch["pad_mask"]
    mean = out["mean"].float()[real]
    y = batch["y"][real]
    parts: Dict[str, float] = {}
    if getattr(cfg, "uncertainty", "gaussian") == "gaussian" and out.get("log_var") is not None:
        log_var = out["log_var"].float()[real].clamp(-10, 10)
        # beta-NLL (cfg.nll_beta, default 0.5): keeps the mean gradient from vanishing when the variance grows
        fit = nll_fn(mean, log_var, y, beta=float(getattr(cfg, "nll_beta", 0.5))).mean()
        parts["nll"] = float(fit.detach())
    else:
        fit = F.mse_loss(mean, y)
        parts["mse"] = float(fit.detach())
    loss = fit
    if flags.get("use_comp", True) and out.get("comp_logits") is not None and "lin" in batch:
        ce = F.cross_entropy(out["comp_logits"].float()[real], batch["lin"][real])
        loss = loss + cfg.lambda_comp * ce
        parts["comp"] = float(ce.detach())
    if flags.get("use_smooth", True) and "nb_pred" in batch and cfg.lambda_smooth > 0:
        # Laplacian smoothness, prediction to prediction: the neighbours' predictions come from the detached
        # cache (written after every step), so only training neighbours count and no target enters the term
        keep = batch["nb_train_mask"][real]                                   # (b, k)
        if keep.any():
            d = (mean[:, None, :] - batch["nb_pred"][real]) ** 2               # (b, k, T)
            sm = d[keep].mean()
            loss = loss + cfg.lambda_smooth * sm
            parts["smooth"] = float(sm.detach())
    lam_align = float(getattr(cfg, "lambda_align_on", 0.1)) if flags.get("align_variant") else float(getattr(cfg, "lambda_align", 0.0))
    if flags.get("use_align", False) and "expr" in batch and out.get("morph_pooled") is not None \
            and lam_align > 0 and real.sum() > 1:
        al = nce_fn(out["morph_pooled"][real], batch["expr"][real])
        loss = loss + lam_align * al
        parts["align"] = float(al.detach())
    parts["total"] = float(loss.detach())
    return loss, parts


# ----------------------------------------------------------------------------- device settings
def configure_backend(cfg: Any, device: str) -> Dict[str, Any]:
    """Apply the single-GPU settings and return what was actually enabled."""
    used = dict(device=device, amp=False, amp_dtype=None, tf32=False, compile=False)
    if device == "cuda":
        if getattr(cfg, "tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            used["tf32"] = True
        if getattr(cfg, "amp_enabled", True):
            used["amp"] = True
            used["amp_dtype"] = getattr(cfg, "amp_dtype", "bfloat16")
        used["device_name"] = torch.cuda.get_device_name(0)
        used["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    used["torch"] = torch.__version__
    say("backend: " + ", ".join(f"{k}={v}" for k, v in used.items()))
    return used


def _autocast(cfg: Any, device: str):
    if device == "cuda" and getattr(cfg, "amp_enabled", True):
        amp_dtype = getattr(cfg, "amp_dtype", "bfloat16")
        if amp_dtype != "bfloat16":
            # float16 needs a GradScaler against gradient underflow and none is implemented here
            raise ValueError(f"amp_dtype={amp_dtype!r} is not supported for training: no GradScaler is implemented, "
                             "so float16 would train unscaled and silently underflow. Use amp_dtype=bfloat16 "
                             "(Ampere, Ada or Hopper GPUs) or amp_enabled=false.")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _forward(model, batch: Dict[str, torch.Tensor], flags: Dict[str, bool]) -> Dict[str, torch.Tensor]:
    return model(batch["centre"], batch.get("grid"), batch["nb"], batch["nb_offsets_um"], batch["sid"], flags)


# ----------------------------------------------------------------------------- training
def train_one(model, batcher_train: CellBatcher, batcher_val: CellBatcher, cfg: Any, device: str,
              flags: Dict[str, bool], seed: int = 0, loader_factory=None, verbose: bool = True
              ) -> Tuple[Any, float, int, pd.DataFrame]:
    """Train ``model`` on the training batcher, early-stop on the validation batcher.

    Returns (model with ``.temperature`` set, best validation fit loss, epochs run, history).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    if getattr(cfg, "compile_model", False) and device == "cuda":
        model = torch.compile(model, mode=getattr(cfg, "compile_mode", "default"), dynamic=False)
        say("model compiled with torch.compile (dynamic=False; batch shapes are static)")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(batcher_train)
    total_steps = max(1, cfg.max_epochs * steps_per_epoch)
    # Warmup is capped at a tenth of the step budget: on a small cohort (few steps per epoch) a fixed
    # 100-step warmup would otherwise consume the whole schedule and the model would never train at peak rate.
    warmup = min(int(getattr(cfg, "warmup_steps", 0)), max(1, total_steps // 10))
    say(f"schedule: {steps_per_epoch} steps/epoch, {total_steps} total steps, warmup {warmup} steps, peak lr {cfg.lr:.1e}, cosine decay")

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    make_loader = loader_factory or (lambda b: iter(b))
    best, best_state, bad, history = float("inf"), None, 0, []
    step = 0
    t_start = time.time()
    cells_seen = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        sums: Dict[str, float] = {}
        n_b = 0
        for batch in make_loader(batcher_train):
            batch = to_device(batch, device)
            with _autocast(cfg, device):
                out = _forward(model, batch, flags)
                loss, parts = batch_loss(out, batch, cfg, flags)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(cfg, "grad_clip", 0) and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            n_b += 1
            # refresh the detached prediction cache the smoothness term reads neighbours from
            real_rows = batch["pad_mask"].cpu().numpy()
            batcher_train.update_cache(batch["idx"].cpu().numpy()[real_rows], out["mean"].detach().float().cpu().numpy()[real_rows])
            cells_seen += int(batch["pad_mask"].sum())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
        val_fit, val_mse = evaluate_fit(model, batcher_val, cfg, device, flags, make_loader)
        row = {f"train_{k}": v / max(1, n_b) for k, v in sums.items()}
        row.update(epoch=epoch, val_fit=val_fit, val_mse=val_mse, lr=opt.param_groups[0]["lr"])
        history.append(row)
        if verbose:
            parts_txt = " ".join(f"{k}={v / max(1, n_b):.4f}" for k, v in sums.items() if k != "total")
            say(f"epoch {epoch:02d} | train total {sums.get('total', 0) / max(1, n_b):.4f} ({parts_txt}) | "
                f"val fit {val_fit:.4f} mse {val_mse:.4f} | lr {row['lr']:.2e}", indent=1)
        if val_fit < best - 1e-4:
            best, bad = val_fit, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                say(f"early stop at epoch {epoch} (no improvement for {cfg.patience} epochs)", indent=1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    elapsed = time.time() - t_start
    model.cells_per_second = cells_seen / max(elapsed, 1e-6)
    say(f"best val fit {best:.4f} after {len(history)} epochs in {elapsed:.1f}s ({model.cells_per_second:.0f} cells/s)")
    model.temperature = 1.0
    if getattr(cfg, "uncertainty", "gaussian") == "gaussian" and getattr(cfg, "temperature_scaling", True):
        model.temperature = fit_temperature(model, batcher_val, cfg, device, flags, make_loader)
    return model, best, len(history), pd.DataFrame(history)


@torch.no_grad()
def evaluate_fit(model, batcher: CellBatcher, cfg: Any, device: str, flags: Dict[str, bool], make_loader=None
                 ) -> Tuple[float, float]:
    """Mean fit loss (NLL or MSE) and plain MSE on the real cells of a batcher."""
    model.eval()
    make_loader = make_loader or (lambda b: iter(b))
    tot_fit, tot_mse, n = 0.0, 0.0, 0
    for batch in make_loader(batcher):
        batch = to_device(batch, device)
        with _autocast(cfg, device):
            out = _forward(model, batch, flags)
        real = batch["pad_mask"]
        mean = out["mean"].float()[real]
        y = batch["y"][real]
        if getattr(cfg, "uncertainty", "gaussian") == "gaussian" and out.get("log_var") is not None:
            fit = gaussian_nll(mean, out["log_var"].float()[real].clamp(-10, 10), y).sum()   # per-element NLL, summed
        else:
            fit = ((mean - y) ** 2).sum()
        tot_fit += float(fit)
        tot_mse += float(((mean - y) ** 2).sum())
        n += int(real.sum()) * y.shape[1]
    return tot_fit / max(n, 1), tot_mse / max(n, 1)


@torch.no_grad()
def fit_temperature(model, batcher: CellBatcher, cfg: Any, device: str, flags: Dict[str, bool], make_loader=None,
                    grid: Optional[Iterable[float]] = None) -> float:
    """Scalar temperature on sigma chosen to minimise the validation NLL (mean unchanged)."""
    make_loader = make_loader or (lambda b: iter(b))
    means, log_vars, ys = [], [], []
    for batch in make_loader(batcher):
        batch = to_device(batch, device)
        with _autocast(cfg, device):
            out = _forward(model, batch, flags)
        real = batch["pad_mask"]
        means.append(out["mean"].float()[real].cpu())
        log_vars.append(out["log_var"].float()[real].clamp(-10, 10).cpu())
        ys.append(batch["y"][real].cpu())
    mean, log_var, y = torch.cat(means), torch.cat(log_vars), torch.cat(ys)
    grid = list(grid) if grid is not None else list(np.exp(np.linspace(np.log(0.25), np.log(4.0), 41)))
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = float(gaussian_nll(mean, log_var + 2 * math.log(t), y).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    say(f"temperature scaling on the validation patient: T = {best_t:.3f} (val NLL {best_nll:.4f})", indent=1)
    return best_t


@torch.no_grad()
def predict(model, batcher: CellBatcher, cfg: Any, device: str, flags: Dict[str, bool], make_loader=None
            ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Out-of-sample prediction: mean, calibrated std, composition probabilities, attention summary.

    Rows are returned in the order of ``batcher.idx``; padded rows are dropped.
    """
    model.eval()
    make_loader = make_loader or (lambda b: iter(b))
    T = float(getattr(model, "temperature", 1.0))
    means, stds, comps, attns, idxs = [], [], [], [], []
    for batch in make_loader(batcher):
        batch = to_device(batch, device)
        with _autocast(cfg, device):
            out = _forward(model, batch, flags)
        real = batch["pad_mask"]
        means.append(out["mean"].float()[real].cpu().numpy())
        if out.get("log_var") is not None:
            stds.append((torch.exp(0.5 * out["log_var"].float()[real].clamp(-10, 10)) * T).cpu().numpy())
        if out.get("comp_logits") is not None:
            comps.append(torch.softmax(out["comp_logits"].float()[real], -1).cpu().numpy())
        if out.get("attn") is not None:
            a = out["attn"]
            attns.append(a.float()[real].cpu().numpy() if torch.is_tensor(a) else np.asarray(a)[real.cpu().numpy()])
        idxs.append(batch["idx"][real].cpu().numpy())
    order = np.concatenate(idxs)
    mean = np.concatenate(means)
    std = np.concatenate(stds) if stds else np.full_like(mean, np.nan)
    comp = np.concatenate(comps) if comps else None
    attn = {"attn": np.concatenate(attns)} if attns else None
    # batcher.idx order: map back so row i corresponds to batcher.idx[i]
    pos = {c: i for i, c in enumerate(batcher.idx)}
    perm = np.array([pos[c] for c in order])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    mean, std = mean[inv], std[inv]
    comp = comp[inv] if comp is not None else None
    if attn is not None:
        attn = {k: v[inv] for k, v in attn.items()}
    return mean, std, comp, attn
