"""
inference_timeseries.py
=======================
Trend-Extraktion fuer eine einzelne synthetische Zeitreihe mittels
reverser Diffusion (DDPM oder DDIM).

Das Skript erzeugt EINE Zeitreihe eines waehlbaren Typs, verrauscht sie
(Forward-Prozess) und extrahiert dann den Trend per Diffusionsmodell.
Das Ergebnis wird als kompakter 3-Panel-Plot gespeichert.

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
    # Basis: periodischer Trend, DDPM, partial-Modus
    python inference_timeseries.py --checkpoint checkpoints/best.pt

    # Chirp mit DDIM, 50 Schritte:
    python inference_timeseries.py --checkpoint checkpoints/best.pt \\
        --trend_type chirp --method ddim --ddim_steps 50

    # Alle neuen Typen in einer Schleife (bash):
    for t in exponential_decay chirp damped_oscillation logistic_growth sawtooth_wave random_walk_trend; do
        python inference_timeseries.py --trend_type $t --output out_$t.png
    done
"""

import argparse
import os
import sys
from pathlib import Path

# Fix module imports by adding parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        description="Trend-Extraktion einer einzelnen Zeitreihe via reverser Diffusion.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Checkpoint & Modell ---
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_0600.pt",
                        help="Pfad zum Modell-Checkpoint")

    # --- Zeitreihe ---
    parser.add_argument(
        "--trend_type", type=str, default="damped_oscillation",
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
    parser.add_argument("--seed",    type=int, default=0,
                        help="Zufallsseed fuer die Zeitreihen-Generierung")

    # --- Inferenz-Modus ---
    parser.add_argument(
        "--inference_mode", type=str, default="partial",
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
    parser.add_argument("--ddim_eta",   type=float, default=0.0,
                        help="DDIM-Stochastizitaet: 0.0=deterministisch, 1.0=wie DDPM")

    # --- Rauschen ---
    parser.add_argument("--noise_level", type=int, default=10,
                        help=(
                            "full-Modus:    wird ignoriert (immer T-1)\n"
                            "partial-Modus: Schritt bei dem der Forward-Prozess stoppt (default: T//2)"
                        ))
    parser.add_argument("--noise_type", type=str, default="mixed",
                        choices=["gaussian", "laplace", "mixed"],
                        help="Art des Rauschens beim Forward-Prozess (nur partial-Modus)")

    # --- Ausgabe ---
    parser.add_argument("--output",      type=str, default="timeseries_inference.png",
                        help="Pfad der Ausgabedatei fuer den Plot")
    parser.add_argument("--output_summary", type=str, default="timeseries_summary.png",
                        help="Pfad der Ausgabedatei fuer den Iterations-Uebersichts-Plot")
    parser.add_argument("--interactive", action="store_true",
                        help="Plot interaktiv anzeigen (plt.show())")
    parser.add_argument("--n_iter", type=int, default=0,
                        help=(
                            "Anzahl weiterer Inferenz-Iterationen, bei denen das Modell\n"
                            "jeweils auf die letzte eigene Vorhersage angewendet wird.\n"
                            "Gesamtzahl Zeilen im Plot: 1 (erste Inferenz) + n_iter."
                        ))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Modell laden (identisch zu inference.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[DiffusionTransformer, dict]:
    """Laedt DiffusionTransformer aus Checkpoint. Gibt (model, saved_args) zurueck."""
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
# Kern-Sampling (identisch zu inference.py)
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
# Hilfsfunktion: eine Zeile im Hauptplot befuellen
# ---------------------------------------------------------------------------

def _fill_row(
    axes_row,
    t_axis:      np.ndarray,
    x_clean_np:  np.ndarray,
    x_input_np:  np.ndarray,
    x_pred_np:   np.ndarray,
    row_label:   str,
    is_first:    bool = False,
) -> None:
    """Befuellt eine Zeile (5 Axes) des Hauptplots."""
    residuum = x_clean_np - x_pred_np
    mse      = float(np.mean(residuum ** 2))
    mae      = float(np.mean(np.abs(residuum)))
    d_clean  = np.gradient(x_clean_np)
    d_pred   = np.gradient(x_pred_np)
    d_resid  = d_clean - d_pred
    d_mse    = float(np.mean(d_resid ** 2))
    d_mae    = float(np.mean(np.abs(d_resid)))

    # Spaltentitel nur in der ersten Zeile
    col_titles = [
        "Eingabe vs. Ground Truth",
        "Vorhersage vs. Ground Truth",
        "Residuum",
        "Ableitung (d/dt)",
        "Ableitungs-Residuum",
    ]
    if is_first:
        for ax, title in zip(axes_row, col_titles):
            ax.set_title(title, fontsize=9, fontweight="bold")

    axes_row[0].set_ylabel(row_label, fontsize=8, fontweight="bold")

    # Panel 1: Eingabe
    axes_row[0].plot(t_axis, x_input_np,  color="tab:orange", lw=0.9, alpha=0.75, label="Eingabe")
    axes_row[0].plot(t_axis, x_clean_np,  color="tab:blue",   lw=1.5, label="Truth")
    axes_row[0].set_ylim(-1.6, 1.6)
    axes_row[0].legend(fontsize=7, loc="upper right")
    axes_row[0].grid(True, alpha=0.3)

    # Panel 2: Vorhersage
    axes_row[1].plot(t_axis, x_clean_np, color="tab:blue", lw=1.5, alpha=0.85, label="Truth")
    axes_row[1].plot(t_axis, x_pred_np,  color="tab:red",  lw=1.5, alpha=0.90,
                     label=f"Pred  MSE={mse:.4f}\nMAE={mae:.4f}")
    axes_row[1].fill_between(t_axis, x_clean_np, x_pred_np, alpha=0.15, color="red")
    axes_row[1].set_ylim(-1.6, 1.6)
    axes_row[1].legend(fontsize=7, loc="upper right")
    axes_row[1].grid(True, alpha=0.3)

    # Panel 3: Residuum
    axes_row[2].bar(t_axis, residuum,
                    color=np.where(residuum >= 0, "tab:blue", "tab:red"),
                    width=1.0, alpha=0.75, linewidth=0)
    axes_row[2].axhline(0, color="black", lw=0.8, linestyle="--")
    axes_row[2].set_ylim(min(-1.0, float(residuum.min()) * 1.15),
                          max( 1.0, float(residuum.max()) * 1.15))
    axes_row[2].grid(True, alpha=0.3)

    # Panel 4: Ableitungen
    axes_row[3].plot(t_axis, d_clean, color="tab:blue", lw=1.5, alpha=0.85, label="d/dt Truth")
    axes_row[3].plot(t_axis, d_pred,  color="tab:red",  lw=1.5, alpha=0.90,
                     label=f"d/dt Pred  MSE={d_mse:.4f}\nMAE={d_mae:.4f}")
    axes_row[3].fill_between(t_axis, d_clean, d_pred, alpha=0.15, color="red")
    axes_row[3].legend(fontsize=7, loc="upper right")
    axes_row[3].grid(True, alpha=0.3)

    # Panel 5: Ableitungs-Residuum
    axes_row[4].bar(t_axis, d_resid,
                    color=np.where(d_resid >= 0, "tab:blue", "tab:red"),
                    width=1.0, alpha=0.75, linewidth=0)
    axes_row[4].axhline(0, color="black", lw=0.8, linestyle="--")
    axes_row[4].set_ylim(min(-1.0, float(d_resid.min()) * 1.15),
                          max( 1.0, float(d_resid.max()) * 1.15))
    axes_row[4].grid(True, alpha=0.3)

    # X-Achsenbeschriftung nur in der letzten Zeile (wird vom Aufrufer gesetzt)


# ---------------------------------------------------------------------------
# Zeitreihen-Plot (Hauptplot, n_iter+1 Zeilen x 5 Spalten)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_timeseries_inference(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    args:      argparse.Namespace,
    device:    torch.device,
) -> list[np.ndarray]:
    """
    Iterative Trend-Extraktion:
      - Erste Zeile: urspruengliche Eingabe → erste Vorhersage
      - Folgezeilen: letzte Vorhersage als neue Eingabe → naechste Vorhersage
      (args.n_iter Wiederholungen nach der ersten Vorhersage)

    Gibt die Liste aller Vorhersagen [x_pred_0, x_pred_1, ...] zurueck
    (Laenge: 1 + n_iter), damit der Summary-Plot diese verwenden kann.

    Jede Zeile hat 5 Spalten:
      Eingabe vs. Truth | Vorhersage vs. Truth | Residuum | Ableitung | Ableitungs-Residuum
    """
    rng = np.random.default_rng(args.seed)

    # --- Ground Truth erzeugen ---
    x_clean_np = generate_trend(args.trend_type, length=args.seq_len, rng=rng)
    x_clean = torch.tensor(x_clean_np, dtype=torch.float32, device=device)
    x_clean = x_clean.unsqueeze(0).unsqueeze(-1)   # (1, L, 1)

    # --- Erste verrauschte Eingabe ---
    t_tensor = torch.full((1,), args.noise_level, device=device, dtype=torch.long)
    x_noisy, _ = scheduler.q_sample(x_clean, t_tensor, noise_type=args.noise_type)
    x_noisy_np  = x_noisy.squeeze().cpu().numpy()

    t_axis      = np.arange(args.seq_len)
    n_rows      = 1 + args.n_iter

    fig, axes = plt.subplots(n_rows, 5, figsize=(25, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # sicherstellen: immer 2D

    fig.suptitle(
        f"Iterative Trend-Extraktion | Typ: {args.trend_type} | "
        f"Modus: {args.inference_mode} | Sampling: {args.method.upper()} | "
        f"Noise-Level: {args.noise_level} | Iterationen: {args.n_iter}",
        fontsize=12, fontweight="bold",
    )

    all_preds: list[np.ndarray] = []

    # Iteration 0: verrauschte Eingabe → erste Vorhersage
    x_input_torch = x_noisy if args.inference_mode == "full" else x_clean
    x_pred_torch  = run_inference(
        model, scheduler, x_input_torch,
        inference_mode=args.inference_mode,
        method=args.method,
        noise_level=args.noise_level,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        noise_type=args.noise_type,
    )
    x_pred_np = x_pred_torch.squeeze().cpu().numpy()
    all_preds.append(x_pred_np)

    _fill_row(axes[0], t_axis, x_clean_np, x_noisy_np, x_pred_np,
              row_label="Iter 0\n(orig. Eingabe)", is_first=True)

    # Iterationen 1 … n_iter: letzte Vorhersage als Eingabe
    for i in range(1, n_rows):
        prev_pred_np = all_preds[-1]
        x_in = torch.tensor(prev_pred_np, dtype=torch.float32, device=device)
        x_in = x_in.unsqueeze(0).unsqueeze(-1)

        x_pred_torch = run_inference(
            model, scheduler, x_in,
            inference_mode=args.inference_mode,
            method=args.method,
            noise_level=args.noise_level,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            noise_type=args.noise_type,
        )
        x_pred_np = x_pred_torch.squeeze().cpu().numpy()
        all_preds.append(x_pred_np)

        _fill_row(axes[i], t_axis, x_clean_np, prev_pred_np, x_pred_np,
                  row_label=f"Iter {i}\n(Pred→Eingabe)", is_first=False)

    # X-Achsenbeschriftung nur in der letzten Zeile
    for ax in axes[-1]:
        ax.set_xlabel("Zeitschritt", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"  Hauptplot gespeichert:    {args.output}")
    if args.interactive:
        plt.show()
    plt.close()

    return x_clean_np, all_preds


# ---------------------------------------------------------------------------
# Summary-Plot: alle Vorhersagen + Ableitungen nebeneinander
# ---------------------------------------------------------------------------

def plot_summary(
    x_clean_np:  np.ndarray,
    all_preds:   list[np.ndarray],
    args:        argparse.Namespace,
) -> None:
    """
    2-spaltiger Uebersichts-Plot:
      Linke Spalte  : Vorhersage (chronologisch untereinander)
      Rechte Spalte : Ableitung der Vorhersage

    Jede Zeile = eine Iteration (Iter 0 oben, Iter n_iter unten).
    Ground Truth ist in jeder Zeile als duenner blauer Hintergrund eingeblendet.
    """
    n_rows  = len(all_preds)
    t_axis  = np.arange(args.seq_len)
    d_clean = np.gradient(x_clean_np)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Iterations-Uebersicht | Typ: {args.trend_type} | "
        f"Modus: {args.inference_mode} | Sampling: {args.method.upper()}",
        fontsize=12, fontweight="bold",
    )

    # Farb-Palette: von blau-gruen nach rot, damit Iterationen unterscheidbar sind
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_rows))

    # Spaltentitel
    axes[0, 0].set_title("Vorhersage", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Ableitung der Vorhersage  (d/dt)", fontsize=10, fontweight="bold")

    for i, (pred_np, color) in enumerate(zip(all_preds, colors)):
        d_pred = np.gradient(pred_np)
        label  = f"Iter {i}" + (" (orig. Eingabe)" if i == 0 else " (Pred→Eingabe)")

        # Linke Spalte: Vorhersage
        ax_l = axes[i, 0]
        ax_l.plot(t_axis, x_clean_np, color="tab:blue", lw=1.0, alpha=0.30, label="Ground Truth")
        ax_l.plot(t_axis, pred_np,    color=color,       lw=1.8, label=label)
        ax_l.fill_between(t_axis, x_clean_np, pred_np, color=color, alpha=0.10)
        ax_l.set_ylim(-1.6, 1.6)
        ax_l.set_ylabel(label, fontsize=8)
        ax_l.legend(fontsize=7, loc="upper right")
        ax_l.grid(True, alpha=0.3)

        # Rechte Spalte: Ableitung
        ax_r = axes[i, 1]
        ax_r.plot(t_axis, d_clean, color="tab:blue", lw=1.0, alpha=0.30, label="d/dt Truth")
        ax_r.plot(t_axis, d_pred,  color=color,       lw=1.8, label=f"d/dt {label}")
        ax_r.fill_between(t_axis, d_clean, d_pred, color=color, alpha=0.10)
        ax_r.legend(fontsize=7, loc="upper right")
        ax_r.grid(True, alpha=0.3)

    # X-Achse nur letzte Zeile
    for ax in axes[-1]:
        ax.set_xlabel("Zeitschritt", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output_summary, dpi=150, bbox_inches="tight")
    print(f"  Summary-Plot gespeichert: {args.output_summary}")
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

    # Default noise_level
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
    print(f"  n_iter         : {args.n_iter}")

    print("\nErstelle iterativen Hauptplot ...")
    x_clean_np, all_preds = plot_timeseries_inference(model, scheduler, args, device)

    print("Erstelle Iterations-Summary-Plot ...")
    plot_summary(x_clean_np, all_preds, args)

    print("\n✓ Fertig.")


if __name__ == "__main__":
    main()

