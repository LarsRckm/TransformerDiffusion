"""
inference.py
============
Trend-Extraktion mittels reverser Diffusion (DDPM und DDIM).

Zwei Inferenz-Modi, wählbar via --inference_mode:

  full      Vollständiger Reverse-Prozess (T → 0).
            Die übergebene verrauschte Zeitreihe wird direkt als x_T
            behandelt. Das Modell iteriert von Schritt T-1 bis 0 und
            "erfindet" dabei einen plausiblen Trend, der zur Eingabe passt.
            → Sinvoll wenn man den Trend bei maximalem Rauschen extrahieren
              möchte oder Conditional Generation erwartet.

  partial   Partieller Reverse-Prozess (t_start → 0, SDEdit-Stil).
            Die Eingabe wird zunächst via Forward-Prozess auf Niveau t_start
            gebracht (zusätzliches Rauschen hinzufügen), und dann ab dort
            denoised. Das bewahrt mehr Struktur aus dem Inputsignal.
            → Sinvoll wenn man den ungefähren Rausch-Level kennt.

Sampling-Methoden (--method):
  ddpm   klassisches stochastisches Sampling (~1000 Schritte)
  ddim   deterministisches Sampling (viel schneller, z.B. 50 Schritte)

Aufruf:
    # Vollständiges Denoising (Start bei t=T-1):
    python inference.py --checkpoint checkpoints/best.pt --inference_mode full

    # Partielles Denoising (Start bei t=500):
    python inference.py --checkpoint checkpoints/best.pt --inference_mode partial --noise_level 500

    # DDIM, schnell:
    python inference.py --checkpoint checkpoints/best.pt --method ddim --ddim_steps 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generator import generate_trend, ALL_TREND_TYPES
from model import DiffusionTransformer
from noise_scheduler import NoiseScheduler, sample_noise


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trend-Extraktion via reverser Diffusion.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Checkpoint & Modell ---
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Pfad zum Modell-Checkpoint")

    # --- Inferenz-Modus ---
    parser.add_argument(
        "--inference_mode", type=str, default="full",
        choices=["full", "partial"],
        help=(
            "full    : Verrauschte Eingabe als x_T → vollständiger Reverse-Prozess T→0\n"
            "partial : Eingabe auf t_start verrauschen (Forward) → Denoising ab t_start→0"
        ),
    )

    # --- Sampling-Methode ---
    parser.add_argument("--method", type=str, default="ddim",
                        choices=["ddpm", "ddim"],
                        help="Sampling-Algorithmus: ddpm (stochastisch) oder ddim (deterministisch)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Anzahl Schritte bei DDIM (ignoriert bei ddpm)")
    parser.add_argument("--ddim_eta",   type=float, default=0.0,
                        help="DDIM-Stochastizität: 0.0=deterministisch, 1.0=wie DDPM")

    # --- Rauschen ---
    parser.add_argument("--noise_level", type=int, default=None,
                        help=(
                            "full-Modus:    wird ignoriert (immer T-1)\n"
                            "partial-Modus: Schritt bei dem der Forward-Prozess stoppt (default: T//2)"
                        ))
    parser.add_argument("--noise_type", type=str, default="mixed",
                        choices=["gaussian", "laplace", "mixed"],
                        help="Art des Rauschens beim Forward-Prozess (nur partial-Modus)")

    # --- Daten ---
    parser.add_argument("--seq_len",    type=int, default=256)
    parser.add_argument("--n_examples", type=int, default=3,
                        help="Anzahl Beispiele pro Trendtyp im Demo-Plot")
    parser.add_argument("--seed",       type=int, default=0)

    # --- Ausgabe ---
    parser.add_argument("--output",      type=str, default="trend_extraction.png")
    parser.add_argument("--interactive", action="store_true",
                        help="Plot interaktiv anzeigen")
    parser.add_argument("--skip_level_analysis", action="store_true",
                        help="Noise-Level-Analyse überspringen")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[DiffusionTransformer, dict]:
    """Lädt DiffusionTransformer aus Checkpoint. Gibt (model, saved_args) zurück."""
    state      = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = state.get("args", {})

    model = DiffusionTransformer(
        seq_len=saved_args.get("seq_len", 256),
        d_model=saved_args.get("d_model", 128),
        nhead=saved_args.get("nhead", 8),
        num_layers=saved_args.get("num_layers", 4),
        dim_feedforward=saved_args.get("dim_ffn", 512),
        time_emb_dim=saved_args.get("time_emb", 128),
        dropout=0.0,
    ).to(device)

    model.load_state_dict(state["model_state"])
    model.eval()

    epoch    = state.get("epoch", "?")
    val_loss = state.get("val_loss", float("nan"))
    print(f"  Checkpoint: Epoche {epoch}, Val-Loss {val_loss:.6f}")
    print(f"  Parameter:  {model.count_parameters():,}")
    return model, saved_args


# ---------------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------------

def robust_normalize(
    y: np.ndarray,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> tuple[np.ndarray, float, float]:
    """
    Normalisiert auf [-1, 1] mittels Quantilen statt Min/Max.
    Robuster gegenüber Rauschen / Ausreißern.

    Returns
    -------
    y_norm : normalisiertes Array
    lo     : unteres Quantil (zum Rückskalieren)
    hi     : oberes Quantil
    """
    lo = float(np.quantile(y, q_low))
    hi = float(np.quantile(y, q_high))
    span = hi - lo if not np.isclose(hi, lo) else 1.0
    y_norm = np.clip(2.0 * (y - lo) / span - 1.0, -1.5, 1.5)
    return y_norm, lo, hi


def denormalize(y_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Kehrt robust_normalize um."""
    span = hi - lo if not np.isclose(hi, lo) else 1.0
    return (y_norm + 1.0) / 2.0 * span + lo


# ---------------------------------------------------------------------------
# Kern-Sampling-Funktion:  wählt full vs. partial
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:          DiffusionTransformer,
    scheduler:      NoiseScheduler,
    x_input:        torch.Tensor,   # (B, L, 1) – normalisiertes (verrauschtes) Signal
    inference_mode: str,            # "full" | "partial"
    method:         str,            # "ddpm" | "ddim"
    noise_level:    int,            # Startschritt (nur für partial)
    ddim_steps:     int = 50,
    ddim_eta:       float = 0.0,
    noise_type:     str = "mixed",
) -> torch.Tensor:
    """
    Führt den Reverse-Diffusions-Prozess durch.

    full-Modus
    ----------
    x_input wird direkt als x_{T-1} behandelt.
    Das Modell denoised von T-1 → 0, ohne Rücksicht auf den exakten
    mathematischen Forward-Prozess des Trainings. Dies entspricht einer
    "Conditional Generation", bei der die Eingabe als Startpunkt dient
    und das Modell einen Trend halluziniert, der zu dieser Eingabe passt.

    partial-Modus (SDEdit-Stil)
    ---------------------------
    1. Forward-Prozess: x_start = q_sample(x_input, t=noise_level, ε)
       Das zusätzliche Rauschen bricht große, strukturfremde Abweichungen auf.
    2. Reverse-Prozess: x_start → x_0
    """
    B = x_input.shape[0]

    if inference_mode == "full":
        # Eingabe direkt als x_T verwenden (kein zusätzlicher Forward-Schritt)
        start_t = scheduler.T - 1
        x_start = x_input

    elif inference_mode == "partial":
        # Forward-Prozess bis noise_level anwenden
        t_tensor = torch.full((B,), noise_level, device=x_input.device, dtype=torch.long)
        x_start, _ = scheduler.q_sample(x_input, t_tensor, noise_type=noise_type)
        start_t = noise_level

    else:
        raise ValueError(f"Unbekannter inference_mode: {inference_mode!r}")

    # Reverse Diffusion
    if method == "ddpm":
        return scheduler.ddpm_sample(model, x_start, start_t=start_t)
    else:  # ddim
        return scheduler.ddim_sample(
            model, x_start,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            start_t=start_t,
        )


# ---------------------------------------------------------------------------
# Demo: Vergleich beider Modi nebeneinander
# ---------------------------------------------------------------------------

@torch.no_grad()
def demo_both_modes(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    args:      argparse.Namespace,
    device:    torch.device,
    T:         int,
):
    """
    Plottet für jeden Trendtyp n_examples Zeilen mit 5 Spalten:
      Spalte 1: Ground Truth (x_0)
      Spalte 2: Verrauschte Eingabe (für partial: x_t; für full: x_T)
      Spalte 3: Ergebnis im 'full'-Modus
      Spalte 4: Ergebnis im 'partial'-Modus
      Spalte 5: Fehler-Overlay (Truth vs. beide Extraktion)
    """
    rng     = np.random.default_rng(args.seed)
    n_types = len(ALL_TREND_TYPES)
    n_ex    = args.n_examples
    partial_t = args.noise_level   # Schritt für partial-Modus

    n_rows = n_types * n_ex
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Ground Truth (x₀)",
        f"Eingabe (verrauscht, t={partial_t} für partial\nbzw. direkt als x_T für full)",
        f"full-Modus\n(Start: x_T)",
        f"partial-Modus\n(Start: x_{partial_t}→0)",
        "Fehler-Overlay",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Vergleich: full vs. partial Inferenz | Sampling: {args.method.upper()}",
        fontsize=13, fontweight="bold",
    )

    row = 0
    t_axis = np.arange(args.seq_len)

    for trend_type in ALL_TREND_TYPES:
        for ex in range(n_ex):
            # --- Sauberen Trend erzeugen ---
            x_clean_np = generate_trend(trend_type, length=args.seq_len, rng=rng)
            x_clean = torch.tensor(x_clean_np, dtype=torch.float32, device=device)
            x_clean = x_clean.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)

            # --- Eingabe: Forward-Prozess bis partial_t (wird für beide gezeigt) ---
            t_tensor = torch.full((1,), partial_t, device=device, dtype=torch.long)
            x_input, _ = scheduler.q_sample(x_clean, t_tensor, noise_type=args.noise_type)
            x_input_np  = x_input.squeeze().cpu().numpy()

            # --- full-Modus: x_input direkt als x_T ---
            x_full = run_inference(
                model, scheduler, x_input,
                inference_mode="full",
                method=args.method,
                noise_level=partial_t,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
            )
            x_full_np = x_full.squeeze().cpu().numpy()

            # --- partial-Modus: Forward auf partial_t, dann reverse ---
            x_partial = run_inference(
                model, scheduler, x_clean,   # geht von x_0 aus!
                inference_mode="partial",
                method=args.method,
                noise_level=partial_t,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                noise_type=args.noise_type,
            )
            x_partial_np = x_partial.squeeze().cpu().numpy()

            mse_full    = float(np.mean((x_clean_np - x_full_np) ** 2))
            mse_partial = float(np.mean((x_clean_np - x_partial_np) ** 2))

            # --- Plotten ---
            def _quick_plot(ax, y, color, lw=1.5, label=None, alpha=1.0):
                ax.plot(t_axis, y, color=color, linewidth=lw, label=label, alpha=alpha)
                ax.set_ylim(-1.5, 1.5)
                ax.grid(True, alpha=0.3)

            # 0: Ground Truth
            _quick_plot(axes[row, 0], x_clean_np, "tab:blue")
            axes[row, 0].set_ylabel(f"{trend_type}\n#{ex+1}", fontsize=8)

            # 1: Verrauschte Eingabe
            _quick_plot(axes[row, 1], x_input_np,  "tab:orange",  lw=0.8, alpha=0.7)
            _quick_plot(axes[row, 1], x_clean_np,  "tab:blue",    lw=1.0, alpha=0.35)

            # 2: full-Modus
            _quick_plot(axes[row, 2], x_clean_np,  "tab:blue",    lw=1.0, alpha=0.35, label="Truth")
            _quick_plot(axes[row, 2], x_full_np,   "tab:purple",  lw=1.5, label=f"MSE={mse_full:.4f}")
            axes[row, 2].legend(fontsize=7, loc="upper right")

            # 3: partial-Modus
            _quick_plot(axes[row, 3], x_clean_np,   "tab:blue",   lw=1.0, alpha=0.35, label="Truth")
            _quick_plot(axes[row, 3], x_partial_np, "tab:green",  lw=1.5, label=f"MSE={mse_partial:.4f}")
            axes[row, 3].legend(fontsize=7, loc="upper right")

            # 4: Fehler-Overlay
            axes[row, 4].plot(t_axis, x_clean_np,   color="tab:blue",   lw=1.5,  label="Truth")
            axes[row, 4].plot(t_axis, x_full_np,    color="tab:purple", lw=1.0,  alpha=0.8, label="full")
            axes[row, 4].plot(t_axis, x_partial_np, color="tab:green",  lw=1.0,  alpha=0.8, label="partial")
            axes[row, 4].fill_between(t_axis, x_clean_np, x_full_np,    alpha=0.15, color="purple")
            axes[row, 4].fill_between(t_axis, x_clean_np, x_partial_np, alpha=0.15, color="green")
            axes[row, 4].set_ylim(-1.5, 1.5)
            axes[row, 4].grid(True, alpha=0.3)
            axes[row, 4].legend(fontsize=7, loc="upper right")

            if row == n_rows - 1:
                for col in range(5):
                    axes[row, col].set_xlabel("Zeitschritt")

            row += 1

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"  Vergleichs-Plot gespeichert: {args.output}")
    if args.interactive:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Noise-Level-Analyse (optional)
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_noise_levels(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    args:      argparse.Namespace,
    device:    torch.device,
    T:         int,
    trend_type: str = "periodic",
):
    """
    Plottet für beide Modi, wie gut die Extraktion bei verschiedenen
    Start-Rausch-Niveaus funktioniert.
    """
    levels  = [int(T * f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
    n       = len(levels)
    rng     = np.random.default_rng(args.seed + 77)
    t_axis  = np.arange(args.seq_len)

    x_clean_np = generate_trend(trend_type, length=args.seq_len, rng=rng)
    x_clean    = torch.tensor(x_clean_np, dtype=torch.float32, device=device)
    x_clean    = x_clean.unsqueeze(0).unsqueeze(-1)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    fig.suptitle(
        f"Noise-Level Analyse ({trend_type}) | full vs. partial | {args.method.upper()}",
        fontsize=12, fontweight="bold",
    )

    row_labels = ["Verrauschte Eingabe", "full-Modus Ergebnis", "partial-Modus Ergebnis"]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=9, fontweight="bold")

    for col, lvl in enumerate(levels):
        t_tensor = torch.full((1,), lvl, device=device, dtype=torch.long)
        x_noisy, _ = scheduler.q_sample(x_clean, t_tensor, noise_type=args.noise_type)
        x_noisy_np  = x_noisy.squeeze().cpu().numpy()

        x_full    = run_inference(
            model, scheduler, x_noisy,
            inference_mode="full",
            method=args.method, noise_level=lvl,
            ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
        )
        x_partial = run_inference(
            model, scheduler, x_clean,
            inference_mode="partial",
            method=args.method, noise_level=lvl,
            ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
            noise_type=args.noise_type,
        )
        x_full_np    = x_full.squeeze().cpu().numpy()
        x_partial_np = x_partial.squeeze().cpu().numpy()

        mse_full    = float(np.mean((x_clean_np - x_full_np) ** 2))
        mse_partial = float(np.mean((x_clean_np - x_partial_np) ** 2))

        snr_db = 10 * np.log10(
            scheduler.alphas_cumprod[lvl].item()
            / (1 - scheduler.alphas_cumprod[lvl].item() + 1e-8)
        )
        axes[0, col].set_title(f"t={lvl}\nSNR={snr_db:.1f} dB", fontsize=9)

        # Zeile 0: Eingabe
        axes[0, col].plot(t_axis, x_noisy_np,  color="tab:orange", lw=0.8)
        axes[0, col].plot(t_axis, x_clean_np,  color="tab:blue",   lw=1.0, alpha=0.4)
        axes[0, col].set_ylim(-2.5, 2.5)

        # Zeile 1: full
        axes[1, col].plot(t_axis, x_clean_np,  color="tab:blue",   lw=1.0, alpha=0.4, label="Truth")
        axes[1, col].plot(t_axis, x_full_np,   color="tab:purple", lw=1.5, label=f"MSE={mse_full:.4f}")
        axes[1, col].set_ylim(-1.5, 1.5)
        axes[1, col].legend(fontsize=7)

        # Zeile 2: partial
        axes[2, col].plot(t_axis, x_clean_np,   color="tab:blue",  lw=1.0, alpha=0.4, label="Truth")
        axes[2, col].plot(t_axis, x_partial_np, color="tab:green", lw=1.5, label=f"MSE={mse_partial:.4f}")
        axes[2, col].set_ylim(-1.5, 1.5)
        axes[2, col].legend(fontsize=7)

        for r in range(3):
            axes[r, col].grid(True, alpha=0.3)
            if r == 2:
                axes[r, col].set_xlabel("Zeitschritt")

    plt.tight_layout()
    save_path = Path(args.output).stem + "_noise_levels.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Noise-Level-Analyse gespeichert: {save_path}")
    if args.interactive:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Public API: extract_trend()
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_trend(
    x_input:        np.ndarray,
    checkpoint:     str,
    inference_mode: str  = "partial",
    method:         str  = "ddim",
    ddim_steps:     int  = 50,
    noise_level:    int  = 500,
    T:              int  = 1000,
    schedule:       str  = "cosine",
    normalize:      bool = True,
    device:         str  = "cpu",
) -> np.ndarray:
    """
    Öffentliche API: Trend-Extraktion aus einer verrauschten Zeitreihe.

    Parameters
    ----------
    x_input        : Verrauschte Zeitreihe (L,) oder (B, L).
    checkpoint     : Pfad zum Modell-Checkpoint.
    inference_mode : "full" | "partial".
    method         : "ddpm" | "ddim".
    ddim_steps     : DDIM-Schritte (nur bei method="ddim").
    noise_level    : Startschritt (nur bei inference_mode="partial").
    T              : Diffusionsschritte.
    schedule       : "linear" | "cosine".
    normalize      : Robuste Quantil-Normalisierung vor der Inferenz anwenden.
    device         : "cpu" | "cuda".

    Returns
    -------
    np.ndarray (L,) oder (B, L) – extrahierter Trend.
    """
    _device = torch.device(device)
    squeeze = x_input.ndim == 1
    if squeeze:
        x_input = x_input[np.newaxis, :]   # (1, L)

    lo_hi_list = []
    xn = x_input.copy()
    if normalize:
        for i in range(xn.shape[0]):
            xn[i], lo, hi = robust_normalize(xn[i])
            lo_hi_list.append((lo, hi))

    x_tensor = torch.tensor(xn, dtype=torch.float32, device=_device).unsqueeze(-1)  # (B,L,1)

    model, _ = load_model(checkpoint, _device)
    sched    = NoiseScheduler(T=T, schedule_type=schedule).to(_device)

    result = run_inference(
        model, sched, x_tensor,
        inference_mode=inference_mode,
        method=method,
        noise_level=noise_level,
        ddim_steps=ddim_steps,
    )

    result_np = result.squeeze(-1).cpu().numpy()  # (B, L)

    # Rückskalierung
    if normalize and lo_hi_list:
        for i, (lo, hi) in enumerate(lo_hi_list):
            result_np[i] = denormalize(result_np[i], lo, hi)

    return result_np[0] if squeeze else result_np


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Checkpoint prüfen
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {ckpt_path}")
        print("  → Starte zuerst das Training: python train.py")
        return

    # Modell & Scheduler laden
    print("Lade Modell ...")
    model, saved_args = load_model(str(ckpt_path), device)

    T        = saved_args.get("T", 1000)
    schedule = saved_args.get("schedule", "cosine")
    seq_len  = saved_args.get("seq_len", args.seq_len)
    args.seq_len = seq_len

    scheduler = NoiseScheduler(T=T, schedule_type=schedule).to(device)

    # Default noise_level
    if args.noise_level is None:
        args.noise_level = T // 2

    print(f"\nInferenz-Konfiguration:")
    print(f"  inference_mode : {args.inference_mode}")
    print(f"  method         : {args.method}")
    if args.method == "ddim":
        print(f"  ddim_steps     : {args.ddim_steps}  (eta={args.ddim_eta})")
    print(f"  noise_level    : {args.noise_level}  (nur partial-Modus)")
    print(f"  seq_len        : {seq_len}")

    # Haupt-Demo: beide Modi nebeneinander
    print("\nErstelle Vergleichs-Demo (full vs. partial) ...")
    demo_both_modes(model, scheduler, args, device, T)

    # Optional: Noise-Level-Analyse
    if not args.skip_level_analysis:
        print("\nNoise-Level-Analyse ...")
        analyze_noise_levels(model, scheduler, args, device, T, trend_type="multi_periodic")

    print("\n✓ Fertig.")


if __name__ == "__main__":
    main()
