"""
create_data.py
==============
Visualisiert die synthetisch generierten Trends und veranschaulicht
den Forward-Process (Rausch-Hinzufügung) für alle vier Trendtypen.

Dieses Skript erzeugt:
  1. trend_examples.png     – Je 4 Beispiele aller 4 Trendtypen
  2. forward_process.png    – Forward-Diffusion bei verschiedenen t-Werten
  3. noise_comparison.png   – Vergleich: Gauß- vs. Laplace-Rauschen
  4. beta_schedule.png      – Linear vs. Cosine Beta-Schedule

Aufruf:
    python create_data.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from data_generator import generate_trend, ALL_TREND_TYPES, TrendDataset
from noise_scheduler import NoiseScheduler, sample_noise


# ---------------------------------------------------------------------------
# Plot 1: Trend-Beispiele
# ---------------------------------------------------------------------------

def plot_trend_examples(n_examples: int = 4, seq_len: int = 256, seed: int = 42):
    """Plottet n_examples Beispiele für jeden der 4 Trendtypen."""
    rng = np.random.default_rng(seed)
    t_axis = np.arange(seq_len)

    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    fig, axes = plt.subplots(n_examples, len(ALL_TREND_TYPES), figsize=(16, 10))
    fig.suptitle("Synthetische Trends – alle Typen (normalisiert auf [-1, 1])",
                 fontsize=14, fontweight="bold", y=1.002)

    type_labels = {
        "slow_trend":    "Slow Trend\nPolynom / Walk+Spline",
        "periodic":      "Periodisch\nsin/cos",
        "multi_periodic": "Multi-Periodisch\nFourier-Stil",
        "discontinuous": "Diskontinuierlich\nSprünge / Knicke",
    }

    for col, ttype in enumerate(ALL_TREND_TYPES):
        for row in range(n_examples):
            trend = generate_trend(ttype, length=seq_len, rng=rng)
            ax = axes[row, col]
            ax.plot(t_axis, trend, color=colors[col], linewidth=1.5)
            ax.set_ylim(-1.3, 1.3)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(type_labels[ttype], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Beispiel {row + 1}", fontsize=9)
            if row == n_examples - 1:
                ax.set_xlabel("Zeitschritt")

    plt.tight_layout()
    plt.savefig("trend_examples.png", dpi=150, bbox_inches="tight")
    print("  ✓ trend_examples.png")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Forward Diffusion Process
# ---------------------------------------------------------------------------

def plot_forward_process(
    seq_len:   int = 256,
    seed:      int = 7,
    T:         int = 1000,
    schedule:  str = "cosine",
    noise_type: str = "mixed",
):
    """
    Visualisiert den Forward-Prozess: Wie sieht x_t für verschiedene t aus?
    """
    scheduler = NoiseScheduler(T=T, schedule_type=schedule)
    rng = np.random.default_rng(seed)
    t_axis = np.arange(seq_len)

    # Alle n_examples * n_levels Plots in einem Grid
    levels     = [0, 100, 250, 500, 750, 999]
    trend_types_to_show = ["periodic", "discontinuous"]
    n_types    = len(trend_types_to_show)
    n_levels   = len(levels)

    fig, axes = plt.subplots(n_types, n_levels, figsize=(3.5 * n_levels, 4 * n_types))
    fig.suptitle(f"Forward Diffusion Process – {schedule.capitalize()} Schedule",
                 fontsize=13, fontweight="bold")

    cmap = plt.cm.plasma
    colors = [cmap(i / (n_levels - 1)) for i in range(n_levels)]

    for row, ttype in enumerate(trend_types_to_show):
        x_clean_np = generate_trend(ttype, length=seq_len, rng=rng)
        x_clean = torch.tensor(x_clean_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for col, t_val in enumerate(levels):
            ax = axes[row, col]

            if t_val == 0:
                y = x_clean_np
                ax.set_title(f"t = 0\n(sauber)", fontsize=9, fontweight="bold")
            else:
                t_tensor = torch.tensor([t_val], dtype=torch.long)
                x_t, _ = scheduler.q_sample(x_clean, t_tensor, noise_type=noise_type)
                y = x_t.squeeze().numpy()
                snr_db = 10 * np.log10(
                    (scheduler.alphas_cumprod[t_val] / (1 - scheduler.alphas_cumprod[t_val])).item()
                )
                ax.set_title(f"t = {t_val}\nSNR = {snr_db:.1f} dB", fontsize=9)

            ax.plot(t_axis, y, color=colors[col], linewidth=1.2)
            ax.plot(t_axis, x_clean_np, color="gray", linewidth=0.8, alpha=0.4, linestyle="--")
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)

            if row == n_types - 1:
                ax.set_xlabel("Zeitschritt")
            if col == 0:
                ax.set_ylabel(ttype, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("forward_process.png", dpi=150, bbox_inches="tight")
    print("  ✓ forward_process.png")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Gauß vs. Laplace Rauschen
# ---------------------------------------------------------------------------

def plot_noise_comparison(seq_len: int = 256, seed: int = 42):
    """Vergleicht Gauß- und Laplace-Rauschen visuell."""
    torch.manual_seed(seed)
    shape = (1, seq_len, 1)

    gaussian = sample_noise(shape, "gaussian", device=torch.device("cpu")).squeeze().numpy()
    laplace  = sample_noise(shape, "laplace",  device=torch.device("cpu")).squeeze().numpy()

    rng = np.random.default_rng(seed)
    x_clean_np = generate_trend("periodic", length=seq_len, rng=rng)

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    t_axis = np.arange(seq_len)

    # --- Zeitreihen: reines Rauschen ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_axis, gaussian, color="steelblue", linewidth=1)
    ax1.set_title("Gauß-Rauschen N(0,1)", fontweight="bold")
    ax1.set_xlabel("Zeitschritt")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_axis, laplace, color="darkorange", linewidth=1)
    ax2.set_title("Laplace-Rauschen L(0, 1/√2)", fontweight="bold")
    ax2.set_xlabel("Zeitschritt")

    # --- Histogramm: Verteilung ---
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-5, 5, 80)
    ax3.hist(gaussian, bins=bins, alpha=0.6, color="steelblue",  label="Gauß",   density=True)
    ax3.hist(laplace,  bins=bins, alpha=0.6, color="darkorange", label="Laplace", density=True)

    # Theoretische Kurven
    x = np.linspace(-5, 5, 500)
    ax3.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="steelblue",  lw=2, label="N(0,1) teoretisch")
    ax3.plot(x, np.sqrt(2)/2 * np.exp(-np.sqrt(2) * np.abs(x)), color="darkorange", lw=2, label="L(0,1/√2) theoretisch")
    ax3.set_title("Verteilungsvergleich", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.set_xlabel("Wert")

    # --- Verrauschte Zeitreihe bei t=500 ---
    scheduler = NoiseScheduler(T=1000, schedule_type="cosine")
    x_tensor = torch.tensor(x_clean_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    t_tensor = torch.tensor([500])

    x_t_gauss_np = scheduler.q_sample(x_tensor, t_tensor, noise_type="gaussian")[0].squeeze().numpy()
    x_t_lap_np   = scheduler.q_sample(x_tensor, t_tensor, noise_type="laplace")[0].squeeze().numpy()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_axis, x_clean_np,   color="gray",       lw=1.5, alpha=0.6, label="Sauber")
    ax4.plot(t_axis, x_t_gauss_np, color="steelblue",  lw=1.0, label="+ Gauß (t=500)")
    ax4.set_title("Periodischer Trend + Gauß", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_ylim(-2.5, 2.5)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_axis, x_clean_np,  color="gray",       lw=1.5, alpha=0.6, label="Sauber")
    ax5.plot(t_axis, x_t_lap_np,  color="darkorange", lw=1.0, label="+ Laplace (t=500)")
    ax5.set_title("Periodischer Trend + Laplace", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.set_ylim(-2.5, 2.5)

    # --- Differenz: Ausreißer durch Laplace ---
    ax6 = fig.add_subplot(gs[1, 2])
    diff = x_t_lap_np - x_t_gauss_np
    ax6.plot(t_axis, diff, color="firebrick", lw=1, label="|Laplace| - |Gauß|")
    ax6.axhline(0, color="gray", lw=0.5)
    ax6.fill_between(t_axis, diff, 0, alpha=0.3, color="firebrick")
    ax6.set_title("Differenz (Laplace - Gauß)\n(Ausreißer-Beitrag)", fontweight="bold")
    ax6.legend(fontsize=8)

    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel("Zeitschritt")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Gauß- vs. Laplace-Rauschen im Diffusionsprozess", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("noise_comparison.png", dpi=150, bbox_inches="tight")
    print("  ✓ noise_comparison.png")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Beta-Schedule
# ---------------------------------------------------------------------------

def plot_beta_schedule(T: int = 1000):
    """Visualisiert linear vs. cosine Beta-Schedule."""
    lin = NoiseScheduler(T=T, schedule_type="linear")
    cos = NoiseScheduler(T=T, schedule_type="cosine")
    ts  = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Beta-Schedules: Linear vs. Cosine", fontsize=13, fontweight="bold")

    # subplot 1: beta_t
    axes[0].plot(ts, lin.betas.numpy(), label="Linear",     color="steelblue")
    axes[0].plot(ts, cos.betas.numpy(), label="Cosine", color="darkorange")
    axes[0].set_title("β_t (Rausch-Varianz pro Schritt)")
    axes[0].set_xlabel("Schritt t")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # subplot 2: alpha_bar_t (Signalstärke)
    axes[1].plot(ts, lin.alphas_cumprod.numpy(), label="Linear",     color="steelblue")
    axes[1].plot(ts, cos.alphas_cumprod.numpy(), label="Cosine", color="darkorange")
    axes[1].set_title("ᾱ_t (Kumulatives Alpha = Signal-Anteil)")
    axes[1].set_xlabel("Schritt t")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # subplot 3: SNR (Signal-to-Noise Ratio)
    lin_snr = lin.alphas_cumprod.numpy() / (1 - lin.alphas_cumprod.numpy() + 1e-8)
    cos_snr = cos.alphas_cumprod.numpy() / (1 - cos.alphas_cumprod.numpy() + 1e-8)
    axes[2].semilogy(ts, lin_snr, label="Linear",     color="steelblue")
    axes[2].semilogy(ts, cos_snr, label="Cosine", color="darkorange")
    axes[2].set_title("SNR = ᾱ_t / (1 - ᾱ_t)  [log-Skala]")
    axes[2].set_xlabel("Schritt t")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("beta_schedule.png", dpi=150, bbox_inches="tight")
    print("  ✓ beta_schedule.png")
    plt.close()


# ---------------------------------------------------------------------------
# Dataset-Statistiken
# ---------------------------------------------------------------------------

def print_dataset_stats(n_samples: int = 1000, seq_len: int = 256):
    """Gibt Statistiken über das erzeugte Dataset aus."""
    print("\n  Dataset-Statistiken:")
    ds = TrendDataset(n_samples=n_samples, seq_len=seq_len, seed=0)
    print(f"  Samples:  {len(ds)}")
    print(f"  Seq len:  {seq_len}")

    for ttype in ALL_TREND_TYPES:
        values = [ds[i]["x_clean"].numpy() for i in range(len(ds)) if ds[i]["trend_type"] == ttype]
        if values:
            all_vals = np.concatenate(values)
            print(f"  {ttype:>15}: n={len(values):4d}, "
                  f"min={all_vals.min():.3f}, max={all_vals.max():.3f}, "
                  f"std={all_vals.std():.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=== create_data.py – Daten-Visualisierung ===\n")
    print("Erstelle Plots ...")

    plot_trend_examples(n_examples=4, seq_len=1000, seed=42)
    plot_forward_process(seq_len=1000, T=1000, schedule="cosine", noise_type="mixed")
    plot_noise_comparison(seq_len=1000, seed=42)
    plot_beta_schedule(T=1000)
    print_dataset_stats(n_samples=1000, seq_len=1000)

    print("\n✓ Alle Plots erstellt:")
    print("  → trend_examples.png")
    print("  → forward_process.png")
    print("  → noise_comparison.png")
    print("  → beta_schedule.png")


if __name__ == "__main__":
    main()
