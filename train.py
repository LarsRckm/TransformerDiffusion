"""
train.py
========
Training-Loop für den DiffusionTransformer.

Features:
  - Mixed-Noise Training (50% Gauß, 50% Laplace)
  - Huber-Loss (robust gegenüber Laplace-Ausreißern)
  - Cosine Annealing LR-Scheduler
  - Checkpoint-Speicherung (bestes Modell + regelmäßige Saves)
  ── Monitoring ─────────────────────────────────────────────
  - Gradient-Norm Tracking pro Epoche
  - Loss aufgeschlüsselt nach t-Bucket (5 Rausch-Niveaus)
  - Denoising-Vorschau alle N Epochen als gespeicherter Plot
  - Abschluss-Plot: Loss, Gradient-Norm, t-Bucket-Loss

Aufruf:
    python train.py
    python train.py --epochs 50 --batch_size 32 --seq_len 1000
    python train.py --preview_every 5 --preview_trend periodic
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data_generator import TrendDataset, ALL_TREND_TYPES, generate_trend
from model import DiffusionTransformer
from noise_scheduler import NoiseScheduler


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training des DiffusionTransformers für Trend-Extraktion."
    )
    # Daten
    parser.add_argument("--n_samples",   type=int,   default=20_000,   help="Datensatz-Größe")
    parser.add_argument("--seq_len",     type=int,   default=1000,     help="Länge der Zeitreihe")
    parser.add_argument("--val_split",   type=float, default=0.1,      help="Anteil Validierungsdaten")
    parser.add_argument("--seed",        type=int,   default=42,       help="Zufallsseed")

    # Modell
    parser.add_argument("--d_model",    type=int,   default=128,      help="d-Dimension")
    parser.add_argument("--nhead",      type=int,   default=8,        help="Attention-Heads")
    parser.add_argument("--num_layers", type=int,   default=4,        help="Transformer-Blöcke")
    parser.add_argument("--dim_ffn",    type=int,   default=512,      help="FFN Hidden-Dim")
    parser.add_argument("--time_emb",   type=int,   default=128,      help="Time-Embedding-Dim")
    parser.add_argument("--dropout",    type=float, default=0.1,      help="Dropout")

    # Noise Scheduler
    parser.add_argument("--T",          type=int,   default=1000,     help="Diffusionsschritte")
    parser.add_argument("--schedule",   type=str,   default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--noise_type", type=str,   default="mixed",  choices=["gaussian", "laplace", "mixed"])
    parser.add_argument("--loss_type",  type=str,   default="huber",  choices=["mse", "huber"])
    parser.add_argument("--smoothness_weight", type=float, default=0.05,
                        help="Gewicht des Glattheitsverlust-Terms (0=aus). Typisch: 0.01-0.1")
    parser.add_argument("--t_bias",     type=str,   default="uniform",
                        choices=["uniform", "low", "verylow"],
                        help=("t-Sampling-Strategie:\n"
                              "  uniform  – gleichverteilt [0, T)  (Standard)\n"
                              "  low      – Bias auf [0, T/4)      (Fine-Tuning)\n"
                              "  verylow  – Bias auf [0, T/10)     (Micro Fine-Tuning)"))

    # Training
    parser.add_argument("--epochs",     type=int,   default=100,      help="Trainings-Epochen")
    parser.add_argument("--batch_size", type=int,   default=64,       help="Batch-Größe")
    parser.add_argument("--lr",         type=float, default=1e-4,     help="Lernrate")
    parser.add_argument("--grad_clip",  type=float, default=1.0,      help="Gradient Clipping (0 = aus)")

    # Ausgabe
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint-Ordner")
    parser.add_argument("--save_every",     type=int, default=10,  help="Checkpoint alle N Epochen")
    parser.add_argument("--resume",         type=str, default=None, help="Checkpoint zum Fortsetzen (checkpoints/best.pt)")

    # Monitoring
    parser.add_argument("--preview_every", type=int, default=10,
                        help="Denoising-Vorschau alle N Epochen (0 = aus)")
    parser.add_argument("--preview_trend", type=str, default="periodic",
                        choices=ALL_TREND_TYPES,
                        help="Trendtyp für die Denoising-Vorschau")
    parser.add_argument("--n_t_buckets",   type=int, default=5,
                        help="Anzahl t-Buckets für die Loss-Aufschlüsselung")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training – eine Epoche
# ---------------------------------------------------------------------------

def _sample_t(B: int, T: int, t_bias: str, device: torch.device) -> torch.Tensor:
    """
    Sampelt Diffusionsschritte t gemäß der gewählten Strategie.

    uniform  : t ~ U[0, T)          – alle Niveau gleich gewichtet
    low      : t ~ U[0, T//4)       – Fokus auf leichtes Rauschen (Fine-Tuning)
    verylow  : t ~ U[0, T//10)      – Fokus auf minimales Rauschen (Micro Fine-Tuning)
    """
    if t_bias == "uniform":
        return torch.randint(0, T, (B,), device=device)
    elif t_bias == "low":
        return torch.randint(0, max(T // 4, 1), (B,), device=device)
    elif t_bias == "verylow":
        return torch.randint(0, max(T // 10, 1), (B,), device=device)
    else:
        raise ValueError(f"Unbekannter t_bias: {t_bias!r}")


def train_one_epoch(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    args:      argparse.Namespace,
) -> tuple[float, float, float, float]:
    """
    Trainiert das Modell für eine Epoche.

    Returns
    -------
    (mean_total_loss, mean_noise_loss, mean_smooth_loss, mean_grad_norm)
    """
    model.train()
    total_loss        = 0.0
    total_noise_loss  = 0.0
    total_smooth_loss = 0.0
    total_grad_norm   = 0.0

    for batch in loader:
        x_clean = batch["x_clean"].to(device).unsqueeze(-1)  # (B, L, 1)
        B = x_clean.shape[0]

        t = _sample_t(B, args.T, args.t_bias, device)

        loss, loss_noise, loss_smooth = scheduler.p_losses(
            model, x_clean, t,
            noise_type=args.noise_type,
            loss_type=args.loss_type,
            smoothness_weight=args.smoothness_weight,
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient-Norm messen (vor dem Clipping)
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        total_loss        += loss.item()
        total_noise_loss  += loss_noise.item()
        total_smooth_loss += loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else float(loss_smooth)
        total_grad_norm   += grad_norm

    n = len(loader)
    return total_loss / n, total_noise_loss / n, total_smooth_loss / n, total_grad_norm / n


# ---------------------------------------------------------------------------
# Validierung – gesamt + pro t-Bucket
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    loader:    DataLoader,
    device:    torch.device,
    args:      argparse.Namespace,
) -> tuple[float, list[float]]:
    """
    Berechnet Validierungs-Loss gesamt und pro t-Bucket.

    Returns
    -------
    (overall_loss, bucket_losses)
        bucket_losses: Liste mit einem Loss-Wert pro Bucket (Länge n_t_buckets).
    """
    model.eval()
    total_loss    = 0.0
    n_buckets     = args.n_t_buckets
    bucket_size   = args.T // n_buckets
    bucket_losses = [0.0] * n_buckets
    bucket_counts = [0]   * n_buckets

    for batch in loader:
        x_clean = batch["x_clean"].to(device).unsqueeze(-1)  # (B, L, 1)
        B = x_clean.shape[0]
        t = _sample_t(B, args.T, args.t_bias, device)

        # Gesamt-Loss
        loss, _, _ = scheduler.p_losses(
            model, x_clean, t,
            noise_type=args.noise_type,
            loss_type=args.loss_type,
            smoothness_weight=args.smoothness_weight,
        )
        total_loss += loss.item()

        # Per-Sample-Loss für Bucket-Analyse
        x_t, noise = scheduler.q_sample(x_clean, t, noise_type=args.noise_type)
        eps_hat     = model(x_t, t)

        if args.loss_type == "mse":
            per_sample = F.mse_loss(eps_hat, noise, reduction="none").mean(dim=(1, 2))
        else:
            per_sample = F.huber_loss(eps_hat, noise, reduction="none", delta=1.0).mean(dim=(1, 2))

        for b_idx in range(n_buckets):
            t_lo = b_idx       * bucket_size
            t_hi = (b_idx + 1) * bucket_size
            mask = (t >= t_lo) & (t < t_hi)
            if mask.any():
                bucket_losses[b_idx] += per_sample[mask].sum().item()
                bucket_counts[b_idx] += mask.sum().item()

    # Bucket-Durchschnitte
    bucket_avgs = [
        (bucket_losses[i] / bucket_counts[i]) if bucket_counts[i] > 0 else 0.0
        for i in range(n_buckets)
    ]

    return total_loss / len(loader), bucket_avgs


# ---------------------------------------------------------------------------
# Denoising-Vorschau
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_denoising_preview(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    epoch:     int,
    args:      argparse.Namespace,
    device:    torch.device,
    ckpt_dir:  Path,
    seed:      int = 999,
):
    """
    Erzeugt eine Denoising-Vorschau für 3 feste Rausch-Niveaus und speichert
    den Plot als PNG im Checkpoint-Ordner.

    Spalten pro Rausch-Level:
        1. Sauberer Trend (Ground Truth)
        2. Verrauschte Eingabe
        3. Extrahierter Trend (DDIM 20 Schritte) + Fehler-Shading
        4. Änderungsrate (1. Ableitung): Ground Truth vs. Extraktion
    """
    model.eval()
    rng = np.random.default_rng(seed)
    x_clean_np = generate_trend(args.preview_trend, length=args.seq_len, rng=rng)
    x_clean    = torch.tensor(x_clean_np, dtype=torch.float32, device=device)
    x_clean    = x_clean.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)

    t_levels     = [
        int(args.T * 0.25),
        int(args.T * 0.50),
        int(args.T * 0.90),
    ]
    level_names  = ["25 %", "50 %", "90 %"]
    n_cols       = 4

    t_axis = np.arange(args.seq_len)
    t_diff = t_axis[:-1]   # x-Achse für die Ableitung (L-1 Punkte)

    fig, axes = plt.subplots(len(t_levels), n_cols,
                             figsize=(5.5 * n_cols, 3.5 * len(t_levels)))
    fig.suptitle(
        f"Denoising-Vorschau – Epoche {epoch + 1} | {args.preview_trend}",
        fontsize=12, fontweight="bold",
    )

    col_labels = [
        "Ground Truth",
        "Verrauscht (Eingabe)",
        "Extrahierter Trend",
        "Änderungsrate (Δ¹)",
    ]
    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(lbl, fontsize=10, fontweight="bold")

    for row, (t_lvl, t_name) in enumerate(zip(t_levels, level_names)):
        # Forward-Prozess
        t_tensor   = torch.full((1,), t_lvl, device=device, dtype=torch.long)
        x_noisy, _ = scheduler.q_sample(x_clean, t_tensor, noise_type="gaussian")

        # Reverse: DDIM (schnell, 20 Schritte)
        x_denoised = scheduler.ddim_sample(
            model, x_noisy, ddim_steps=20, eta=0.0, start_t=t_lvl
        )

        x_c  = x_clean.squeeze().cpu().numpy()
        x_n  = x_noisy.squeeze().cpu().numpy()
        x_d  = x_denoised.squeeze().cpu().numpy()
        mse  = float(np.mean((x_c - x_d) ** 2))

        # 1. Ableitungen (finite Differenz)
        d_truth = np.diff(x_c)   # (L-1,)
        d_pred  = np.diff(x_d)   # (L-1,)

        # --- Spalte 0: Ground Truth ---
        axes[row, 0].plot(t_axis, x_c, color="tab:blue", lw=1.5)
        axes[row, 0].set_ylabel(f"t={t_lvl}\n({t_name})", fontsize=9)

        # --- Spalte 1: Verrauscht ---
        axes[row, 1].plot(t_axis, x_n, color="tab:orange", lw=0.8, alpha=0.7)
        axes[row, 1].plot(t_axis, x_c, color="tab:blue",   lw=1.0, alpha=0.3)

        # --- Spalte 2: Extraktion ---
        axes[row, 2].plot(t_axis, x_c, color="tab:blue",  lw=1.0, alpha=0.4, label="Truth")
        axes[row, 2].plot(t_axis, x_d, color="tab:green", lw=1.5, label=f"MSE={mse:.5f}")
        axes[row, 2].fill_between(t_axis, x_c, x_d, alpha=0.2, color="red")
        axes[row, 2].legend(fontsize=7, loc="upper right")

        # --- Spalte 3: Ableitung ---
        axes[row, 3].plot(t_diff, d_truth, color="tab:blue",  lw=1.0, alpha=0.7, label="Truth Δ¹")
        axes[row, 3].plot(t_diff, d_pred,  color="tab:green", lw=1.2, alpha=0.9, label="Pred Δ¹")
        axes[row, 3].axhline(0, color="gray", lw=0.5, ls="--")
        axes[row, 3].fill_between(t_diff, d_truth, d_pred, alpha=0.15, color="red")
        axes[row, 3].legend(fontsize=7, loc="upper right")

        # y-Achsen setzen
        d_max = max(np.abs(d_truth).max(), np.abs(d_pred).max()) * 1.3 + 1e-4
        axes[row, 3].set_ylim(-d_max, d_max)

        for col in range(n_cols):
            axes[row, col].set_ylim(-1.5, 1.5) if col < 3 else None
            axes[row, col].grid(True, alpha=0.3)
            if row == len(t_levels) - 1:
                axes[row, col].set_xlabel("Zeitschritt")

    plt.tight_layout()
    out_path = ckpt_dir / f"preview_epoch_{epoch + 1:04d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  🖼  Vorschau gespeichert: {out_path.name}")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     DiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    args:      argparse.Namespace,
    path:      Path,
):
    state = {
        "epoch":       epoch,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "args":        vars(args),
    }
    torch.save(state, path)


def load_checkpoint(
    path:      str,
    model:     DiffusionTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optim_state" in state:
        optimizer.load_state_dict(state["optim_state"])
    print(f"  ✓ Checkpoint geladen: Epoche {state['epoch']}, Val-Loss {state['val_loss']:.6f}")
    return state["epoch"], state["val_loss"]


# ---------------------------------------------------------------------------
# Abschluss-Plot
# ---------------------------------------------------------------------------

def plot_training_summary(
    train_losses:  list[float],
    val_losses:    list[float],
    grad_norms:    list[float],
    bucket_losses: list[list[float]],   # [epoch][bucket]
    args:          argparse.Namespace,
    save_path:     Path,
):
    """
    Erstellt einen 3-Panel-Abschlussplot:
      1. Train- und Val-Loss
      2. Gradient-Norm (vor Clipping)
      3. Val-Loss aufgeschlüsselt nach t-Bucket
    """
    epochs   = range(1, len(train_losses) + 1)
    n_b      = args.n_t_buckets
    bucket_sz = args.T // n_b

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training-Zusammenfassung – DiffusionTransformer",
                 fontsize=13, fontweight="bold")

    # Panel 1: Loss
    axes[0].plot(epochs, train_losses, label="Train",      color="steelblue",   lw=1.5)
    axes[0].plot(epochs, val_losses,   label="Validation", color="darkorange",  lw=1.5)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoche")
    axes[0].set_ylabel(f"{args.loss_type.capitalize()}-Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Panel 2: Gradient-Norm
    axes[1].plot(epochs, grad_norms, color="firebrick", lw=1.5)
    if args.grad_clip > 0:
        axes[1].axhline(args.grad_clip, color="gray", ls="--", lw=1,
                        label=f"Clip-Grenze ({args.grad_clip})")
        axes[1].legend(fontsize=9)
    axes[1].set_title("Gradient-Norm (vor Clipping)")
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("L2-Norm")
    axes[1].grid(True, alpha=0.4)

    # Panel 3: Loss pro t-Bucket
    bucket_arr = np.array(bucket_losses)   # (n_epochs, n_buckets)
    cmap       = plt.cm.plasma
    for b in range(n_b):
        t_lo  = b       * bucket_sz
        t_hi  = (b + 1) * bucket_sz
        color = cmap(b / max(n_b - 1, 1))
        axes[2].plot(epochs, bucket_arr[:, b],
                     label=f"t=[{t_lo}, {t_hi})",
                     color=color, lw=1.5)
    axes[2].set_title("Val-Loss je Rausch-Level (t-Bucket)")
    axes[2].set_xlabel("Epoche")
    axes[2].set_ylabel(f"{args.loss_type.capitalize()}-Loss")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Zusammenfassungs-Plot gespeichert: {save_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    # --- Datensatz ---
    print(f"\nErzeuge Datensatz: {args.n_samples} Samples, seq_len={args.seq_len} ...")
    dataset    = TrendDataset(n_samples=args.n_samples, seq_len=args.seq_len, seed=args.seed)
    val_size   = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    print(f"  Train: {train_size} | Val: {val_size}")

    # --- Modell ---
    model = DiffusionTransformer(
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ffn,
        time_emb_dim=args.time_emb,
        dropout=args.dropout,
    ).to(device)
    print(f"\nModell: {model.count_parameters():,} trainierbare Parameter")

    # --- Noise Scheduler ---
    scheduler = NoiseScheduler(T=args.T, schedule_type=args.schedule).to(device)

    # --- Optimizer + LR-Scheduler ---
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Checkpoint fortsetzen ---
    start_epoch   = 0
    best_val_loss = math.inf
    if args.resume is not None:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    # --- Training ---
    print(f"\nStarte Training für {args.epochs} Epochen ...\n")

    train_losses:  list[float]       = []
    val_losses:    list[float]       = []
    grad_norms:    list[float]       = []
    bucket_losses: list[list[float]] = []

    bucket_headers = " | ".join(
        f"t=[{b * (args.T // args.n_t_buckets)}, {(b+1) * (args.T // args.n_t_buckets)})"
        for b in range(args.n_t_buckets)
    )
    print(f"{'Epoche':>8} | {'Train':>10} | {'L_noise':>10} | {'L_smooth':>10} | {'Val':>10} | {'GradNorm':>9} | {bucket_headers}")
    print("-" * (70 + 20 * args.n_t_buckets))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_noise, train_smooth, grad_norm = train_one_epoch(
            model, scheduler, train_loader, optimizer, device, args
        )
        val_loss, b_losses = validate(model, scheduler, val_loader, device, args)
        lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        grad_norms.append(grad_norm)
        bucket_losses.append(b_losses)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        bucket_str = " | ".join(f"{bl:10.6f}" for bl in b_losses)
        print(
            f"{epoch+1:8d} | {train_loss:10.6f} | {train_noise:10.6f} | {train_smooth:10.6f} | "
            f"{val_loss:10.6f} | {grad_norm:9.4f} | {bucket_str}  [{elapsed:.1f}s, lr={lr_now:.1e}]"
        )

        # Bestes Modell
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, args, ckpt_dir / "best.pt")
            print(f"  ★ Bestes Modell gespeichert (Val-Loss: {val_loss:.6f})")

        # Regelmäßige Checkpoints
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, args,
                            ckpt_dir / f"epoch_{epoch+1:04d}.pt")

        # Denoising-Vorschau
        if args.preview_every > 0 and (epoch + 1) % args.preview_every == 0:
            save_denoising_preview(model, scheduler, epoch, args, device, ckpt_dir)

    # --- Abschluss ---
    plot_training_summary(
        train_losses, val_losses, grad_norms, bucket_losses,
        args, ckpt_dir / "training_summary.png",
    )
    print(f"\n✓ Training abgeschlossen. Bestes Val-Loss: {best_val_loss:.6f}")
    print(f"  Checkpoint:  {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
