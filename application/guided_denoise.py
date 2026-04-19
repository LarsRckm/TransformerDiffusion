"""
guided_denoise.py
=================
Trend-Extraktion via "Guided Diffusion" (ähnlich RePaint):

  1. Start bei t = T-1 mit PUREM RAUSCHEN (kein Signal-Bias).
  2. Bei jedem Reverse-Schritt t → t-1 wird die verrauschte Eingabe
     auf den Schritt t-1 re-verrauscht und mit dem Modell-Output gemischt:

        x_{t-1} = blend * x_{t-1}^{model}  +  (1-blend) * x_{t-1}^{input}

     Damit "verankert" die verrauschte Eingabe die grobe Form, ohne dass
     das Modell falsch konditioniert wird (kein Trainings-Mismatch).

  3. Der Blend-Faktor kann entweder konstant oder zeitabhängig sein:
       - konstant:    blend = blend_strength   (für alle Schritte gleich)
       - linear:      blend wächst von 0 → 1  je näher t an 0 kommt
                      → Eingabe dominiert bei hohem Rauschen,
                        Modell verfeinert bei niedrigem Rauschen (empfohlen)
       - cosine:      sanfte S-Kurve, empirisch gut

Vorteile gegenüber `full`-Modus:
  - Kein Trainings-Mismatch: Modell sieht immer korrekt re-verrauschte Eingabe
  - Kein Retraining nötig
  - Kontrollierbar über --blend_strength und --blend_schedule

Beispiel-Aufruf:
    python guided_denoise.py --checkpoint checkpoints/best.pt

    python guided_denoise.py --checkpoint checkpoints/best.pt \\
        --trend_type chirp --blend_schedule cosine --blend_strength 0.5 \\
        --noise_level 200
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
        description="Guided Trend-Extraktion via RePaint-Blending.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_0600.pt")

    # Zeitreihe
    parser.add_argument(
        "--trend_type", type=str, default="periodic",
        choices=ALL_TREND_TYPES,
        help="Typ der synthetischen Zeitreihe.",
    )
    parser.add_argument("--seq_len",     type=int,   default=1000)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--noise_level", type=int,   default=150,
                        help="Forward-Schritt für die verrauschte Eingabe\n"
                             "(wie stark die Eingabe verrauscht wird, bevor sie\n"
                             " als Form-Anker genutzt wird).")
    parser.add_argument("--noise_type",  type=str,   default="mixed",
                        choices=["gaussian", "laplace", "mixed"])

    # Guided Sampling
    parser.add_argument(
        "--blend_strength", type=float, default=0.5,
        help=(
            "Maximale Modell-Gewichtung in der Mischung  [0.0 … 1.0].\n"
            "  0.0 → nur Eingabe (keine Glättung)\n"
            "  1.0 → nur Modell  (wie normales Denoising)\n"
            "  0.5 → gleichgewichtig (empfohlen)"
        ),
    )
    parser.add_argument(
        "--blend_schedule", type=str, default="cosine",
        choices=["constant", "linear", "cosine"],
        help=(
            "Wie sich der Blend-Faktor über die Zeitschritte verändert:\n"
            "  constant : immer blend_strength (gleichmäßig)\n"
            "  linear   : wächst von 0 → blend_strength  (Eingabe dominiert zuerst)\n"
            "  cosine   : S-Kurve, sanfterer Übergang (empfohlen)"
        ),
    )

    # Sampling Methode
    parser.add_argument("--method", type=str, default="ddpm",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int,   default=200)
    parser.add_argument("--ddim_eta",   type=float, default=0.0)

    # Output
    parser.add_argument("--output",      type=str, default="guided_denoise.png")
    parser.add_argument("--no-interactive", dest="interactive", action="store_false")
    parser.set_defaults(interactive=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[DiffusionTransformer, dict]:
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
# Blend-Faktor berechnen
# ---------------------------------------------------------------------------

def compute_blend(t: int, T: int, blend_strength: float, blend_schedule: str) -> float:
    """
    Gibt den Modell-Blend-Faktor λ für Schritt t zurück.

    λ=1 → reines Modell-Output
    λ=0 → reines Eingabe-Rauschen

    Mit fortschreitendem Denoising (t von T→0) soll das Modell mehr
    Einfluss gewinnen (verfeinern), während die Eingabe zuerst die Form
    vorgibt.

    Parameters
    ----------
    t              : Aktueller Schritt (T-1 … 0)
    T              : Gesamtanzahl Schritte
    blend_strength : Maximale Modell-Gewichtung [0, 1]
    blend_schedule : "constant", "linear" oder "cosine"

    Returns
    -------
    λ ∈ [0, blend_strength]
    """
    # Normierter Fortschritt: 0 = Anfang (hoher Rausch), 1 = Ende (sauber)
    progress = 1.0 - t / (T - 1)  # 0 bei t=T-1, 1 bei t=0

    if blend_schedule == "constant":
        return blend_strength
    elif blend_schedule == "linear":
        return blend_strength * progress
    elif blend_schedule == "cosine":
        # S-Kurve: langsam starts, schnell mitte, langsam ende
        return blend_strength * (1 - np.cos(np.pi * progress)) / 2.0
    else:
        raise ValueError(f"Unbekannter blend_schedule: {blend_schedule!r}")


# ---------------------------------------------------------------------------
# Guided DDPM Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_ddpm_sample(
    model:          DiffusionTransformer,
    scheduler:      NoiseScheduler,
    x_input:        torch.Tensor,   # (B, L, 1) – Original-Eingabe (sauber oder leicht verrauscht)
    x_noisy:        torch.Tensor,   # (B, L, 1) – Eingabe auf noise_level verrauscht
    noise_level:    int,
    blend_strength: float,
    blend_schedule: str,
) -> tuple[torch.Tensor, list[float]]:
    """
    Guided DDPM-Sampling (RePaint-Stil):

    - Start: pures Rauschen (x_T ~ N(0,1))
    - Bei jedem Schritt t -> t-1:
        1. Normaler DDPM-Schritt: x_{t-1}^model
        2. Eingabe auf t-1 re-verrauschen: x_{t-1}^input
        3. Blenden: x_{t-1} = λ*x_{t-1}^model + (1-λ)*x_{t-1}^input

    Parameters
    ----------
    model          : Trainiertes DiffusionTransformer-Modell.
    scheduler      : NoiseScheduler-Instanz.
    x_input        : Saubere (oder leicht verrauschte) Eingabe-Zeitreihe (B, L, 1).
    x_noisy        : Eingabe bei t=noise_level (für Visualisierung).
    noise_level    : Bis zu welchem Rauschpegel die Eingabe verrauscht wird
                     (bestimmt wie stark der Formeinfluss ist).
    blend_strength : Maximale Modell-Gewichtung [0, 1].
    blend_schedule : Art des Blend-Faktor-Verlaufs.

    Returns
    -------
    x_0         : Denoised Zeitreihe (B, L, 1).
    blend_curve : Liste der Blend-Faktoren über alle Schritte (für Plot).
    """
    T = scheduler.T
    B = x_input.shape[0]

    # Start: pures Rauschen (korrekte x_T Verteilung)
    x = torch.randn_like(x_input)

    blend_curve: list[float] = []

    for t in reversed(range(0, T)):
        t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)

        # ----- Normaler DDPM-Step -----
        eps_hat      = model(x, t_tensor)
        sqrt_recip_a = scheduler._extract(scheduler.sqrt_recip_alphas,    t_tensor, x.shape)
        beta_t       = scheduler._extract(scheduler.betas,                 t_tensor, x.shape)
        sqrt_one_minus = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)

        x_model = sqrt_recip_a * (x - beta_t / sqrt_one_minus * eps_hat)

        if t > 0:
            posterior_var = scheduler._extract(scheduler.posterior_variance, t_tensor, x.shape)
            x_model = x_model + torch.sqrt(posterior_var) * torch.randn_like(x)

        # ----- Eingabe auf t-1 re-verrauschen (nur solange t-1 <= noise_level) -----
        lam = compute_blend(t, T, blend_strength, blend_schedule)
        blend_curve.append(lam)

        if t > 0 and (t - 1) <= noise_level:
            # Eingabe auf t-1 re-verrauschen: konsistente Verteilung
            t_prev_tensor = torch.full((B,), t - 1, device=x.device, dtype=torch.long)
            x_input_t_prev, _ = scheduler.q_sample(x_input, t_prev_tensor)

            # Blenden: Modell verfeinert die grobe Form der Eingabe
            x = lam * x_model + (1.0 - lam) * x_input_t_prev
        else:
            # Bei t=0 oder wenn t-1 > noise_level: nur Modell
            x = x_model

    return x, blend_curve


# ---------------------------------------------------------------------------
# Guided DDIM Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_ddim_sample(
    model:          DiffusionTransformer,
    scheduler:      NoiseScheduler,
    x_input:        torch.Tensor,
    noise_level:    int,
    ddim_steps:     int,
    eta:            float,
    blend_strength: float,
    blend_schedule: str,
) -> tuple[torch.Tensor, list[float]]:
    """
    Guided DDIM-Sampling (RePaint-Stil, deterministisch).
    """
    T = scheduler.T
    B = x_input.shape[0]

    # Start: pures Rauschen
    x = torch.randn_like(x_input)

    # Gleichmäßig verteilte Zeitschritte
    timesteps = torch.linspace(T - 1, 0, ddim_steps + 1).long().tolist()
    blend_curve: list[float] = []

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1]

        t_tensor = torch.full((B,), t_curr, device=x.device, dtype=torch.long)

        a_t    = scheduler.alphas_cumprod[t_curr]
        a_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        eps_hat = model(x, t_tensor)

        # x_0 schätzen
        x0_pred = (x - torch.sqrt(1 - a_t) * eps_hat) / torch.sqrt(a_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # DDIM-Schritt Rauschen
        sigma = (
            eta
            * torch.sqrt((1 - a_prev) / (1 - a_t))
            * torch.sqrt(1 - a_t / a_prev)
        )
        noise   = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x_model = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev - sigma**2) * eps_hat + sigma * noise

        # ----- Guided Blending -----
        lam = compute_blend(t_curr, T, blend_strength, blend_schedule)
        blend_curve.append(lam)

        if t_prev >= 0 and t_prev <= noise_level:
            t_prev_tensor = torch.full((B,), max(t_prev, 0), device=x.device, dtype=torch.long)
            x_input_t_prev, _ = scheduler.q_sample(x_input, t_prev_tensor)
            x = lam * x_model + (1.0 - lam) * x_input_t_prev
        else:
            x = x_model

    return x, blend_curve


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_and_plot(
    model:     DiffusionTransformer,
    scheduler: NoiseScheduler,
    args:      argparse.Namespace,
    device:    torch.device,
) -> None:
    rng = np.random.default_rng(args.seed)

    # Ground Truth erzeugen
    x_clean_np = generate_trend(args.trend_type, length=args.seq_len, rng=rng)
    x_clean    = torch.tensor(x_clean_np, dtype=torch.float32, device=device)
    x_clean_t  = x_clean.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)

    # Verrauschte Eingabe (Forward-Prozess) – für Visualisierung & als Form-Anker
    t_tensor  = torch.full((1,), args.noise_level, device=device, dtype=torch.long)
    x_noisy_t, _ = scheduler.q_sample(x_clean_t, t_tensor, noise_type=args.noise_type)
    x_noisy_np = x_noisy_t.squeeze().cpu().numpy()

    # ----- Guided Sampling -----
    print(f"  Starte Guided Sampling ({args.method.upper()}, blend={args.blend_schedule}, "
          f"strength={args.blend_strength}) ...")

    if args.method == "ddpm":
        x_pred_t, blend_curve = guided_ddpm_sample(
            model, scheduler, x_clean_t, x_noisy_t,
            noise_level    = args.noise_level,
            blend_strength = args.blend_strength,
            blend_schedule = args.blend_schedule,
        )
    else:
        x_pred_t, blend_curve = guided_ddim_sample(
            model, scheduler, x_clean_t,
            noise_level    = args.noise_level,
            ddim_steps     = args.ddim_steps,
            eta            = args.ddim_eta,
            blend_strength = args.blend_strength,
            blend_schedule = args.blend_schedule,
        )

    x_pred_np = x_pred_t.squeeze().cpu().numpy()

    # ----- Referenz: normales partial Sampling -----
    print("  Referenz: partial Sampling ...")
    t_ref = torch.full((1,), args.noise_level, device=device, dtype=torch.long)
    x_ref_in, _ = scheduler.q_sample(x_clean_t, t_ref, noise_type=args.noise_type)
    x_ref_np = scheduler.ddpm_sample(model, x_ref_in, start_t=args.noise_level).squeeze().cpu().numpy()

    # Metriken
    t_axis   = np.arange(args.seq_len)
    res_guided = x_clean_np - x_pred_np
    res_ref    = x_clean_np - x_ref_np
    mse_g  = float(np.mean(res_guided ** 2))
    mae_g  = float(np.mean(np.abs(res_guided)))
    mse_r  = float(np.mean(res_ref   ** 2))
    mae_r  = float(np.mean(np.abs(res_ref)))

    # ----- Plot -----
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Guided Denoising  |  Typ: {args.trend_type}  |  Noise-Level: {args.noise_level}  |  "
        f"Blend: {args.blend_schedule} (strength={args.blend_strength})  |  "
        f"Methode: {args.method.upper()}",
        fontsize=13, fontweight="bold",
    )

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    # --- Zeile 1: Eingabe ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Eingabe  vs.  Ground Truth", fontsize=10, fontweight="bold")
    ax1.plot(t_axis, x_noisy_np, color="tab:orange", lw=0.9, alpha=0.7,
             label=f"Verrauscht (t={args.noise_level})")
    ax1.plot(t_axis, x_clean_np, color="tab:blue",   lw=1.8, label="Ground Truth")
    ax1.set_ylim(-1.8, 1.8)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Zeile 2: Guided vs Referenz ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title(f"Guided Vorhersage  (MSE={mse_g:.4f}, MAE={mae_g:.4f})",
                  fontsize=10, fontweight="bold")
    ax2.plot(t_axis, x_clean_np, color="tab:blue", lw=1.2, alpha=0.40, label="Ground Truth")
    ax2.plot(t_axis, x_pred_np,  color="tab:red",  lw=1.8, label="Guided")
    ax2.fill_between(t_axis, x_clean_np, x_pred_np, alpha=0.15, color="red")
    ax2.set_ylim(-1.8, 1.8)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Zeitschritt", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(f"Referenz: partial Sampling  (MSE={mse_r:.4f}, MAE={mae_r:.4f})",
                  fontsize=10, fontweight="bold")
    ax3.plot(t_axis, x_clean_np, color="tab:blue",  lw=1.2, alpha=0.40, label="Ground Truth")
    ax3.plot(t_axis, x_ref_np,   color="tab:green", lw=1.8, label="Partial")
    ax3.fill_between(t_axis, x_clean_np, x_ref_np, alpha=0.15, color="green")
    ax3.set_ylim(-1.8, 1.8)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Zeitschritt", fontsize=9)

    # --- Zeile 3: Residua & Blend-Kurve ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_title("Residuum: Guided vs. Partial", fontsize=10, fontweight="bold")
    ax4.plot(t_axis, res_guided, color="tab:red",   lw=1.2, alpha=0.85, label="Guided")
    ax4.plot(t_axis, res_ref,    color="tab:green", lw=1.2, alpha=0.85, label="Partial")
    ax4.axhline(0, color="black", lw=0.8, ls="--")
    ax4.legend(fontsize=9, loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Zeitschritt", fontsize=9)
    ax4.set_ylabel("Ground Truth − Vorhersage", fontsize=8)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title("Blend-Faktor  λ(t)  über Zeitschritte", fontsize=10, fontweight="bold")
    steps = np.arange(len(blend_curve))
    ax5.plot(steps, blend_curve, color="tab:purple", lw=1.5)
    ax5.fill_between(steps, 0, blend_curve, alpha=0.15, color="tab:purple")
    ax5.axhline(args.blend_strength, color="gray", lw=0.9, ls="--",
                label=f"Max = {args.blend_strength}")
    ax5.set_ylim(-0.05, args.blend_strength * 1.15 + 0.05)
    ax5.set_ylabel("λ  (Modell-Anteil)", fontsize=9)
    ax5.set_xlabel("Sampling-Schritt (0=Start / reines Rauschen)", fontsize=8)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n  Plot gespeichert: {args.output}")
    if args.interactive:
        plt.show()
    plt.close()

    # Zusammenfassung
    print(f"\n  Guided   │ MSE={mse_g:.5f}  MAE={mae_g:.5f}")
    print(f"  Partial  │ MSE={mse_r:.5f}  MAE={mae_r:.5f}")
    winner = "Guided" if mse_g < mse_r else "Partial"
    print(f"  Besser   → {winner}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {ckpt_path}")
        return

    print("Lade Modell ...")
    model, saved_args = load_model(str(ckpt_path), device)

    T        = saved_args.get("T", 1000)
    schedule = saved_args.get("schedule", "cosine")
    seq_len  = saved_args.get("seq_len", args.seq_len)
    args.seq_len = seq_len

    scheduler = NoiseScheduler(T=T, schedule_type=schedule).to(device)

    print(f"\nKonfiguration:")
    print(f"  trend_type     : {args.trend_type}")
    print(f"  noise_level    : {args.noise_level}")
    print(f"  noise_type     : {args.noise_type}")
    print(f"  method         : {args.method}")
    if args.method == "ddim":
        print(f"  ddim_steps     : {args.ddim_steps}  (eta={args.ddim_eta})")
    print(f"  blend_strength : {args.blend_strength}")
    print(f"  blend_schedule : {args.blend_schedule}")
    print(f"  seq_len        : {seq_len}")
    print(f"  seed           : {args.seed}")

    print("\nStarte Guided Denoising ...")
    run_and_plot(model, scheduler, args, device)
    print("\n✓ Fertig.")


if __name__ == "__main__":
    main()
