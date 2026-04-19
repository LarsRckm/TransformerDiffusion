"""
denoise_timeseries.py
=====================
Entrauscht eine einzelne synthetische Zeitreihe mittels reverser Diffusion
(DDPM oder DDIM) und visualisiert das Ergebnis in einem kompakten 3x2-
Spalten-Plot sowie einem separaten FFT-Differenz-Plot.

Plot-Layout (Hauptplot, 3 Spalten x 2 Zeilen):
  Linke Spalte   : Ground Truth (oben) | Modellvorhersage (unten)
  Mittlere Spalte: Ableitung von Ground Truth (oben) | Ableitung von Vorhersage (unten)
  Rechte Spalte  : FFT von Ground Truth (oben) | FFT von Vorhersage (unten)

Separater Plot:
  Differenz der beiden FFT-Amplitudenspektren (|FFT(truth)| - |FFT(pred)|)

Unterstuetzte Trend-Typen (--trend_type):
  Basis (aus dem Training):
    slow_trend, periodic, multi_periodic, discontinuous
  Erweitert (neue Funktionstypen):
    exponential_decay, chirp, damped_oscillation,
    logistic_growth, sawtooth_wave, random_walk_trend

Inferenz-Modi (--inference_mode):
  full      Verrauschte Eingabe wird direkt als x_T behandelt.
            Das Modell denoised von T-1 bis 0.
  partial   Eingabe wird via Forward-Prozess auf t_start verrauscht
            (SDEdit-Stil) und dann denoised.

Sampling-Methoden (--method):
  ddpm   klassisches stochastisches Sampling (~1000 Schritte)
  ddim   deterministisches Sampling (schneller, z.B. 50 Schritte)

Beispiel-Aufrufe:
    python denoise_timeseries.py --checkpoint checkpoints/best.pt

    python denoise_timeseries.py --checkpoint checkpoints/best.pt \\
        --trend_type chirp --method ddim --ddim_steps 50 --noise_level 200
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
        description="Entrauschung einer einzelnen Zeitreihe via reverser Diffusion + FFT-Analyse.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Checkpoint & Modell ---
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_0600.pt",
                        help="Pfad zum Modell-Checkpoint")

    # --- Zeitreihe ---
    parser.add_argument(
        "--trend_type", type=str, default="periodic",
        choices=ALL_TREND_TYPES,
        help=(
            "Typ der synthetischen Zeitreihe. Basis-Typen:\n"
            "  slow_trend, periodic, multi_periodic, discontinuous\n"
            "Erweiterte Typen (nicht im Training gesehen):\n"
            "  exponential_decay  – Exponentieller Abfall/Anstieg\n"
            "  chirp              – Frequenz-Sweep (Sinuswelle mit steigender Frequenz)\n"
            "  damped_oscillation – Gedaempfte Sinusschwingung\n"
            "  logistic_growth    – S-foermige Wachstumskurve (Sigmoid)\n"
            "  sawtooth_wave      – Saegezahn / Dreieckswelle\n"
            "  random_walk_trend  – Gefilterte Brownsche Bewegung"
        ),
    )
    parser.add_argument("--seq_len", type=int, default=1000,
                        help="Laenge der Zeitreihe (wird aus Checkpoint uebernommen wenn moeglich)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Zufallsseed fuer die Zeitreihen-Generierung")

    # --- Inferenz-Modus ---
    parser.add_argument(
        "--inference_mode", type=str, default="full",
        choices=["full", "partial"],
        help=(
            "full    : Verrauschte Eingabe als x_T → vollstaendiger Reverse-Prozess T→0\n"
            "partial : Eingabe auf t_start verrauschen (Forward) → Denoising ab t_start→0"
        ),
    )

    # --- Sampling-Methode ---
    parser.add_argument("--method", type=str, default="ddpm",
                        choices=["ddpm", "ddim"],
                        help="Sampling-Algorithmus: ddpm (stochastisch) oder ddim (deterministisch)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Anzahl Schritte bei DDIM (ignoriert bei ddpm)")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM-Stochastizitaet: 0.0=deterministisch, 1.0=wie DDPM")

    # --- Rauschen ---
    parser.add_argument("--noise_level", type=int, default=150,
                        help=(
                            "full-Modus:    wird ignoriert (immer T-1)\n"
                            "partial-Modus: Schritt bei dem der Forward-Prozess stoppt (default: 10)"
                        ))
    parser.add_argument("--noise_type", type=str, default="mixed",
                        choices=["gaussian", "laplace", "mixed"],
                        help="Art des Rauschens beim Forward-Prozess (nur partial-Modus)")

    # --- Ausgabe ---
    parser.add_argument("--output", type=str, default="denoise_main.png",
                        help="Pfad der Ausgabedatei fuer den 3x2-Hauptplot")
    parser.add_argument("--output_fft_diff", type=str, default="denoise_fft_diff.png",
                        help="Pfad der Ausgabedatei fuer den separaten FFT-Differenz-Plot")
    parser.add_argument("--no-interactive", dest="interactive", action="store_false",
                        help="plt.show() unterdruecken (z.B. fuer Batch-Laeufe)")
    parser.set_defaults(interactive=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[DiffusionTransformer, dict]:
    """Laedt DiffusionTransformer aus Checkpoint. Gibt (model, saved_args) zurueck."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    epoch = state.get("epoch", "?")
    val_loss = state.get("val_loss", float("nan"))
    print(f"  Checkpoint: Epoche {epoch}, Val-Loss {val_loss:.6f}")
    print(f"  Parameter:  {model.count_parameters():,}")
    return model, saved_args


# ---------------------------------------------------------------------------
# Kern-Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:          DiffusionTransformer,
    scheduler:      NoiseScheduler,
    x_input:        torch.Tensor,   # (B, L, 1)
    inference_mode: str,
    method:         str,
    noise_level:    int,
    ddim_steps:     int = 50,
    ddim_eta:       float = 0.0,
    noise_type:     str = "mixed",
) -> torch.Tensor:
    """Fuehrt den Reverse-Diffusions-Prozess durch (full oder partial)."""
    B = x_input.shape[0]

    if inference_mode == "full":
        start_t = scheduler.T - 1
        x_start = x_input
    elif inference_mode == "partial":
        t_tensor = torch.full((B,), noise_level, device=x_input.device, dtype=torch.long)
        x_start, _ = scheduler.q_sample(x_input, t_tensor, noise_type=noise_type)
        start_t = noise_level
    else:
        raise ValueError(f"Unbekannter inference_mode: {inference_mode!r}")

    if method == "ddpm":
        return scheduler.ddpm_sample(model, x_start, start_t=start_t)
    else:
        return scheduler.ddim_sample(
            model, x_start,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            start_t=start_t,
        )


# ---------------------------------------------------------------------------
# FFT-Hilfsfunktion
# ---------------------------------------------------------------------------

def compute_fft(signal: np.ndarray, sample_rate: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet einseitiges Amplitudenspektrum via FFT.

    Gibt (freqs, amplitudes) zurueck. Nur positive Frequenzen (0 … Nyquist).
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    amplitudes = np.abs(fft_vals) / N          # normiert auf Signalamplitude
    amplitudes[1:-1] *= 2                       # einseitig → doppelte Amplitude
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    return freqs, amplitudes


# ---------------------------------------------------------------------------
# Haupt-Plot-Funktion (3 Spalten × 2 Zeilen)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_denoise(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    args:      argparse.Namespace,
    device:    torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt den 3x2-Hauptplot:
      Linke Spalte   : Ground Truth (oben) | Vorhersage (unten)
      Mittlere Spalte: Ableitung Ground Truth (oben) | Ableitung Vorhersage (unten)
      Rechte Spalte  : FFT Ground Truth (oben) | FFT Vorhersage (unten)

    Gibt (x_clean_np, x_pred_np) fuer den separaten FFT-Differenz-Plot zurueck.
    """
    rng = np.random.default_rng(args.seed)

    # --- Ground Truth erzeugen ---
    x_clean_np = generate_trend(args.trend_type, length=args.seq_len, rng=rng)
    x_clean = torch.tensor(x_clean_np, dtype=torch.float32, device=device)
    x_clean_t = x_clean.unsqueeze(0).unsqueeze(-1)   # (1, L, 1)

    # --- Verrauschte Eingabe (Forward-Prozess) ---
    t_tensor = torch.full((1,), args.noise_level, device=device, dtype=torch.long)
    x_noisy_t, _ = scheduler.q_sample(x_clean_t, t_tensor, noise_type=args.noise_type)
    x_noisy_np = x_noisy_t.squeeze().cpu().numpy()

    # --- Vorhersage (Reverse-Prozess ab noise_level) ---
    x_pred_torch = run_inference(
        model, scheduler, x_clean_t,
        inference_mode=args.inference_mode,
        method=args.method,
        noise_level=args.noise_level,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        noise_type=args.noise_type,
    )
    x_pred_np = x_pred_torch.squeeze().cpu().numpy()

    t_axis = np.arange(args.seq_len)

    # --- Ableitungen ---
    d_clean = np.gradient(x_clean_np)
    d_pred  = np.gradient(x_pred_np)

    # --- FFT ---
    freqs_clean, amp_clean = compute_fft(x_clean_np)
    freqs_pred,  amp_pred  = compute_fft(x_pred_np)

    # --- Metriken ---
    residuum = x_clean_np - x_pred_np
    mse = float(np.mean(residuum ** 2))
    mae = float(np.mean(np.abs(residuum)))

    d_resid = d_clean - d_pred
    d_mse = float(np.mean(d_resid ** 2))
    d_mae = float(np.mean(np.abs(d_resid)))

    # -----------------------------------------------------------------------
    # Hauptplot: 3 Spalten × 2 Zeilen
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    fig.suptitle(
        f"Zeitreihen-Entrauschung | Typ: {args.trend_type} | "
        f"Modus: {args.inference_mode} | Sampling: {args.method.upper()} | "
        f"Noise-Level: {args.noise_level} | Seed: {args.seed}",
        fontsize=12, fontweight="bold",
    )

    col_titles = ["Zeitreihe", "Ableitung  (d/dt)", "FFT-Amplitudenspektrum"]
    row_labels  = ["Ground Truth", f"Modellvorhersage  (MSE={mse:.4f}, MAE={mae:.4f})"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    # -----------------------------------------------------------------------
    # Linke Spalte (col 0): Zeitreihe
    # -----------------------------------------------------------------------
    # Oben: Ground Truth + verrauschte Eingabe
    ax = axes[0, 0]
    ax.plot(t_axis, x_noisy_np, color="tab:orange", lw=0.8, alpha=0.65, label=f"Verrauscht (t={args.noise_level})")
    ax.plot(t_axis, x_clean_np, color="tab:blue",   lw=1.5, label="Ground Truth")
    ax.set_ylim(-1.6, 1.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Unten: Vorhersage + verrauschte Eingabe
    ax = axes[1, 0]
    ax.plot(t_axis, x_noisy_np, color="tab:orange", lw=0.8, alpha=0.65, label=f"Verrauscht (t={args.noise_level})")
    ax.plot(t_axis, x_clean_np, color="tab:blue",   lw=1.0, alpha=0.35, label="Ground Truth")
    ax.plot(t_axis, x_pred_np,  color="tab:red",    lw=1.5, label="Vorhersage")
    ax.fill_between(t_axis, x_clean_np, x_pred_np, alpha=0.15, color="red")
    ax.set_ylim(-1.6, 1.6)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Zeitschritt", fontsize=8)

    # -----------------------------------------------------------------------
    # Mittlere Spalte (col 1): Ableitung
    # -----------------------------------------------------------------------
    # Oben: Ableitung Ground Truth
    ax = axes[0, 1]
    ax.plot(t_axis, d_clean, color="tab:blue", lw=1.5, label="d/dt  Ground Truth")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Unten: Ableitung Vorhersage
    ax = axes[1, 1]
    ax.plot(t_axis, d_clean, color="tab:blue",  lw=1.0, alpha=0.35, label="d/dt  Ground Truth")
    ax.plot(t_axis, d_pred,  color="tab:orange", lw=1.5,
            label=f"d/dt  Vorhersage  (MSE={d_mse:.4f}, MAE={d_mae:.4f})")
    ax.fill_between(t_axis, d_clean, d_pred, alpha=0.15, color="orange")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Zeitschritt", fontsize=8)

    # -----------------------------------------------------------------------
    # Rechte Spalte (col 2): FFT
    # -----------------------------------------------------------------------
    # Oben: FFT Ground Truth
    ax = axes[0, 2]
    ax.plot(freqs_clean, amp_clean, color="tab:blue", lw=1.2, label="FFT  Ground Truth")
    ax.set_xlabel("Frequenz", fontsize=8)
    ax.set_ylabel("Amplitude (normiert)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Unten: FFT Vorhersage
    ax = axes[1, 2]
    ax.plot(freqs_clean, amp_clean, color="tab:blue", lw=1.0, alpha=0.35, label="FFT  Ground Truth")
    ax.plot(freqs_pred,  amp_pred,  color="tab:red",  lw=1.2, label="FFT  Vorhersage")
    ax.set_xlabel("Frequenz", fontsize=8)
    ax.set_ylabel("Amplitude (normiert)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"  Hauptplot gespeichert: {args.output}")
    if args.interactive:
        plt.show()
    plt.close()

    return x_clean_np, x_pred_np


# ---------------------------------------------------------------------------
# Separater FFT-Differenz-Plot
# ---------------------------------------------------------------------------

def plot_fft_diff(
    x_clean_np: np.ndarray,
    x_pred_np:  np.ndarray,
    args:       argparse.Namespace,
) -> None:
    """
    Separater Plot: Differenz des FFT-Amplitudenspektrums.
      |FFT(truth)| - |FFT(pred)|

    Positive Werte: Ground Truth hat mehr Energie bei dieser Frequenz.
    Negative Werte: Vorhersage hat mehr Energie bei dieser Frequenz.
    """
    freqs_clean, amp_clean = compute_fft(x_clean_np)
    freqs_pred,  amp_pred  = compute_fft(x_pred_np)

    # Frequenzachsen sind identisch (gleiche Signallaenge), aber sicherheitshalber:
    min_len = min(len(amp_clean), len(amp_pred))
    freqs  = freqs_clean[:min_len]
    diff   = amp_clean[:min_len] - amp_pred[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    fig.suptitle(
        f"FFT-Differenzanalyse | Typ: {args.trend_type} | "
        f"Modus: {args.inference_mode} | Sampling: {args.method.upper()} | "
        f"Noise-Level: {args.noise_level} | Seed: {args.seed}",
        fontsize=12, fontweight="bold",
    )

    # Oben: beide Spektren überlagert
    ax_top = axes[0]
    ax_top.plot(freqs, amp_clean, color="tab:blue", lw=1.5, label="FFT  Ground Truth", alpha=0.85)
    ax_top.plot(freqs, amp_pred,  color="tab:red",  lw=1.5, label="FFT  Vorhersage",   alpha=0.85)
    ax_top.set_title("FFT-Amplitudenspektren im Vergleich", fontsize=10, fontweight="bold")
    ax_top.set_ylabel("Amplitude (normiert)", fontsize=9)
    ax_top.legend(fontsize=9, loc="upper right")
    ax_top.grid(True, alpha=0.3)

    # Unten: Differenz als Balkendiagramm
    ax_bot = axes[1]
    colors = np.where(diff >= 0, "tab:blue", "tab:red")
    ax_bot.bar(freqs, diff, width=(freqs[1] - freqs[0]) * 0.9,
               color=colors, alpha=0.80, linewidth=0)
    ax_bot.axhline(0, color="black", lw=0.9, linestyle="--")
    ax_bot.set_title(
        "Differenz  |FFT(Truth)| − |FFT(Vorhersage)|  "
        "(blau = Truth dominiert, rot = Vorhersage dominiert)",
        fontsize=10, fontweight="bold",
    )
    ax_bot.set_xlabel("Frequenz", fontsize=9)
    ax_bot.set_ylabel("Amplitudendifferenz", fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    # Energie-Statistik als Text-Box
    total_energy_clean = float(np.sum(amp_clean ** 2))
    total_energy_pred  = float(np.sum(amp_pred  ** 2))
    max_diff_freq      = freqs[np.argmax(np.abs(diff))]
    textstr = (
        f"Energie Truth:     {total_energy_clean:.4f}\n"
        f"Energie Vorhersage:{total_energy_pred:.4f}\n"
        f"Max. |Δ| bei f =   {max_diff_freq:.4f}"
    )
    ax_bot.text(
        0.98, 0.97, textstr,
        transform=ax_bot.transAxes,
        fontsize=8, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    plt.tight_layout()
    plt.savefig(args.output_fft_diff, dpi=150, bbox_inches="tight")
    print(f"  FFT-Differenz-Plot gespeichert: {args.output_fft_diff}")
    if args.interactive:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Checkpoint pruefen
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

    # Default noise_level (falls nicht gesetzt)
    if args.noise_level is None:
        args.noise_level = T // 2

    print(f"\nKonfiguration:")
    print(f"  trend_type     : {args.trend_type}")
    print(f"  inference_mode : {args.inference_mode}")
    print(f"  method         : {args.method}")
    if args.method == "ddim":
        print(f"  ddim_steps     : {args.ddim_steps}  (eta={args.ddim_eta})")
    print(f"  noise_level    : {args.noise_level}")
    print(f"  noise_type     : {args.noise_type}")
    print(f"  seq_len        : {seq_len}")
    print(f"  seed           : {args.seed}")

    print("\nErstelle 3×2-Hauptplot ...")
    x_clean_np, x_pred_np = plot_denoise(model, scheduler, args, device)

    print("Erstelle FFT-Differenz-Plot ...")
    plot_fft_diff(x_clean_np, x_pred_np, args)

    print("\n✓ Fertig.")


if __name__ == "__main__":
    main()
