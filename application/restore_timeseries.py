"""restore_timeseries.py
======================

Conditional restoration / imputation for time series.

Features:
  - works with missing values via a mask (1=observed, 0=missing)
  - estimates unknown measurement noise sigma via SigmaEstimator
  - deterministic default (DDIM, eta=0, fixed seed)
  - optional multi-sample inference for uncertainty bands
  - overlap-add stitching for long sequences (window=1000, overlap=200)

Input format:
  - Provide a 1D .npy file with shape (N,) containing the observed series.
  - Missing values can be encoded as NaN.

Output:
  - Saves restored series as .npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Fix module imports by adding parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DiffusionTransformer, SigmaEstimator
from noise_scheduler import NoiseScheduler
from normalization import robust_normalize_masked, denormalize, apply_norm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional denoise/impute time series")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Path to input .npy (NaN allowed)")
    p.add_argument("--output", type=str, default="restored.npy")

    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=200)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--method", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1)

    return p.parse_args()


def _hann_weights(L: int) -> np.ndarray:
    if L <= 1:
        return np.ones(L, dtype=np.float32)
    return np.hanning(L).astype(np.float32)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DiffusionTransformer, SigmaEstimator, dict]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = state.get("args", {})

    model = DiffusionTransformer(
        seq_len=saved_args.get("seq_len", 1000),
        d_model=saved_args.get("d_model", 128),
        nhead=saved_args.get("nhead", 8),
        num_layers=saved_args.get("num_layers", 4),
        dim_feedforward=saved_args.get("dim_ffn", 512),
        time_emb_dim=saved_args.get("time_emb", 128),
        dropout=0.0,
        in_channels=3,
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    sigma_est = SigmaEstimator(hidden=32).to(device)
    if "sigma_state" in state:
        sigma_est.load_state_dict(state["sigma_state"], strict=True)
    sigma_est.eval()

    return model, sigma_est, saved_args


@torch.no_grad()
def restore_window(
    y_win: np.ndarray,
    model: DiffusionTransformer,
    sigma_est: SigmaEstimator,
    sched: NoiseScheduler,
    *,
    method: str,
    ddim_steps: int,
    ddim_eta: float,
    seed: int,
    n_samples: int,
) -> np.ndarray:
    """Restore a single window. Returns (n_samples, L) if n_samples>1 else (L,)."""
    L = y_win.shape[0]
    mask = (~np.isnan(y_win)).astype(np.float32)
    y_filled = np.nan_to_num(y_win, nan=0.0).astype(np.float32)

    # normalize based on observed points
    y_norm, params = robust_normalize_masked(y_filled, mask)
    y_obs_norm = (y_norm * mask).astype(np.float32)

    device = next(model.parameters()).device
    y_t = torch.tensor(y_obs_norm, device=device).view(1, L, 1)
    m_t = torch.tensor(mask, device=device).view(1, L, 1)

    # estimate sigma once in normalized space
    sigma_log_hat = sigma_est(y_t, m_t)  # (1,)

    outs = []
    for k in range(n_samples):
        torch.manual_seed(int(seed) + k)
        # Start from noise; conditional sampling uses y/m only via model.
        x = torch.randn_like(y_t)

        if method == "ddpm":
            for t in reversed(range(0, sched.T)):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                # DDPM step using conditional eps prediction
                eps_hat = model(x, t_tensor, y_obs=y_t, mask=m_t, sigma_log=sigma_log_hat)
                sqrt_recip_a = sched._extract(sched.sqrt_recip_alphas, t_tensor, x.shape)
                beta_t = sched._extract(sched.betas, t_tensor, x.shape)
                sqrt_one_minus = sched._extract(sched.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
                model_mean = sqrt_recip_a * (x - beta_t / sqrt_one_minus * eps_hat)
                if t == 0:
                    x = model_mean
                else:
                    posterior_var = sched._extract(sched.posterior_variance, t_tensor, x.shape)
                    x = model_mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            # DDIM
            timesteps = torch.linspace(sched.T - 1, 0, ddim_steps + 1).long().tolist()
            x_curr = x
            for i in range(len(timesteps) - 1):
                t_curr = timesteps[i]
                t_prev = timesteps[i + 1]
                t_tensor = torch.full((1,), t_curr, device=device, dtype=torch.long)

                a_t = sched.alphas_cumprod[t_curr]
                a_prev = sched.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

                eps_hat = model(x_curr, t_tensor, y_obs=y_t, mask=m_t, sigma_log=sigma_log_hat)
                x0_pred = (x_curr - torch.sqrt(1 - a_t) * eps_hat) / torch.sqrt(a_t)
                x0_pred = torch.clamp(x0_pred, -1.5, 1.5)

                sigma = (
                    ddim_eta
                    * torch.sqrt((1 - a_prev) / (1 - a_t))
                    * torch.sqrt(1 - a_t / a_prev)
                )
                noise = torch.randn_like(x_curr) if ddim_eta > 0 else torch.zeros_like(x_curr)
                x_curr = (
                    torch.sqrt(a_prev) * x0_pred
                    + torch.sqrt(1 - a_prev - sigma**2) * eps_hat
                    + sigma * noise
                )
            x = x_curr

        x_np = x.squeeze(0).squeeze(-1).cpu().numpy()
        out = denormalize(x_np, params)
        outs.append(out.astype(np.float32))

    outs = np.stack(outs, axis=0)
    return outs[0] if n_samples == 1 else outs


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = np.load(args.input).astype(np.float32)
    N = y.shape[0]
    W = int(args.window)
    O = int(args.overlap)
    if W <= 0 or O < 0 or O >= W:
        raise ValueError("Invalid window/overlap")
    stride = W - O

    model, sigma_est, saved_args = load_model(args.checkpoint, device)
    T = int(saved_args.get("T", args.T))
    schedule = str(saved_args.get("schedule", args.schedule))
    sched = NoiseScheduler(T=T, schedule_type=schedule).to(device)

    w = _hann_weights(W)
    sum_w = np.zeros(N, dtype=np.float32)

    if args.n_samples == 1:
        sum_pred = np.zeros(N, dtype=np.float32)
    else:
        sum_pred = np.zeros((args.n_samples, N), dtype=np.float32)

    # pad to cover tail
    n_windows = int(np.ceil(max(N - W, 0) / stride)) + 1
    for wi in range(n_windows):
        start = wi * stride
        end = start + W
        if end <= N:
            y_win = y[start:end]
            out = restore_window(
                y_win,
                model,
                sigma_est,
                sched,
                method=args.method,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                seed=args.seed + wi * 1000,
                n_samples=args.n_samples,
            )
            if args.n_samples == 1:
                sum_pred[start:end] += w * out
            else:
                sum_pred[:, start:end] += w[None, :] * out
            sum_w[start:end] += w
        else:
            # last partial: pad with NaN
            pad = end - N
            y_win = np.concatenate([y[start:N], np.full(pad, np.nan, dtype=np.float32)], axis=0)
            out = restore_window(
                y_win,
                model,
                sigma_est,
                sched,
                method=args.method,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                seed=args.seed + wi * 1000,
                n_samples=args.n_samples,
            )
            out = out[..., : (N - start)]
            ww = w[: (N - start)]
            if args.n_samples == 1:
                sum_pred[start:N] += ww * out
            else:
                sum_pred[:, start:N] += ww[None, :] * out
            sum_w[start:N] += ww

    sum_w = np.clip(sum_w, 1e-8, None)
    if args.n_samples == 1:
        pred = sum_pred / sum_w
    else:
        pred = sum_pred / sum_w[None, :]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pred)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
