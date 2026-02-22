"""
data_generator.py
=================
Synthetischer Daten-Generator für Trendextraktion mit Diffusion-Modellen.

Erzeugt vier Trend-Typen:
  - slow_trend:       Langsam ändernde Trends (Polynom niedriger Ordnung ODER
                      iterativer Random-Walk + Spline-Glättung)
  - periodic:         Einzelne Sinus/Cosinus-Welle
  - multi_periodic:   Überlagerung mehrerer Sinuswellen (Fourier-Stil)
  - discontinuous:    Trends mit Sprüngen oder plötzlichen Steigungsänderungen

Alle Trends werden auf [-1, 1] normalisiert, bevor Rauschen hinzugefügt wird.
"""

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from torch.utils.data import Dataset
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Trend-Generierungsfunktionen
# ---------------------------------------------------------------------------

TrendType = Literal["slow_trend", "periodic", "multi_periodic", "discontinuous"]


def generate_slow_trend(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Erzeugt einen langsam ändernden Trend über zwei Modi (zufällig gewählt):

    Modus A – Polynom niedriger Ordnung (Grad 1–3):
        Zufällige Koeffizienten für ein Polynom f(t) = a0 + a1*t + a2*t² + ...
        mit kleinen Koeffizienten, damit sich der Trend nicht zu schnell ändert.

    Modus B – Iterativer Random-Walk + Spline-Glättung:
        1. In jedem Schritt wird ein kleiner Wert aus [-step, +step] addiert.
        2. Anschließend wird eine univariate Spline-Funktion (scipy) angepasst,
           die den Verlauf stark glättet – sodass ein organisch wirkender,
           kontinuierlicher Trend entsteht.
    """
    t = np.linspace(0.0, 1.0, length)
    mode = rng.choice(["polynomial", "walk"])

    if mode == "polynomial":
        # Grad zufällig zwischen 1 und 3
        degree = rng.integers(1, 4)
        # Koeffizienten klein halten → langsame Änderung
        coeffs = rng.uniform(-1.5, 1.5, size=degree + 1)
        # Höhere Koeffizienten stärker dämpfen
        for k in range(2, degree + 1):
            coeffs[k] *= 0.3 ** (k - 1)
        signal = np.polyval(coeffs[::-1], t)   # np.polyval erwartet höchste Potenz zuerst
        # alternativ direkt auswerten:
        signal = sum(coeffs[k] * t**k for k in range(degree + 1))

    else:  # mode == "walk"
        # Schrittweite aus festgelegtem Bereich
        step_scale = rng.uniform(0.005, 0.03)   # kleine Schritte → langsame Änderung
        steps = rng.uniform(-step_scale, step_scale, size=length)
        signal = np.cumsum(steps)

        # Univariate Spline zur Glättung: wenige Knoten → sehr glatt
        x_idx = np.linspace(0.0, 1.0, length)
        # s = Smoothing-Parameter; größer → stärker geglättet
        s_factor = rng.uniform(length * 0.05, length * 0.2)
        try:
            spline = UnivariateSpline(x_idx, signal, s=s_factor, k=3)
            signal = spline(x_idx)
        except Exception:
            # Fallback: gleitender Mittelwert
            w = max(length // 10, 5)
            signal = np.convolve(signal, np.ones(w) / w, mode="same")

    return signal


def generate_periodic(length: int, rng: np.random.Generator) -> np.ndarray:
    """Einzelne Sinus/Cosinus-Welle mit zufälliger Freq., Amplitude & Phase."""
    t = np.linspace(0.0, 2 * np.pi, length)
    freq = rng.uniform(1.0, 5.0)
    amp  = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0.0, 2 * np.pi)
    # Mische manchmal sin und cos
    use_cos = rng.random() > 0.5
    if use_cos:
        return amp * np.cos(freq * t + phase)
    else:
        return amp * np.sin(freq * t + phase)


def generate_multi_periodic(length: int, rng: np.random.Generator) -> np.ndarray:
    """Überlagerung von 2–4 Sinuswellen (Fourier-Stil)."""
    t = np.linspace(0.0, 2 * np.pi, length)
    n_components = rng.integers(2, 5)  # 2, 3 oder 4 Komponenten
    signal = np.zeros(length)
    for _ in range(n_components):
        freq  = rng.uniform(1.0, 8.0)
        amp   = rng.uniform(0.2, 1.5)
        phase = rng.uniform(0.0, 2 * np.pi)
        signal += amp * np.sin(freq * t + phase)
    return signal


def generate_discontinuous(length: int, rng: np.random.Generator) -> np.ndarray:
    """Trend mit Sprüngen (step functions) oder plötzlichen Steigungsänderungen."""
    t = np.linspace(0.0, 1.0, length)
    signal = np.zeros(length)

    # Basis: leichter linearer Trend
    a = rng.uniform(-1.0, 1.0)
    signal += a * t

    disc_type = rng.choice(["step", "slope_change", "mixed"])

    n_events = rng.integers(1, 4)  # 1–3 Diskontinuitäten

    for _ in range(n_events):
        pos = rng.integers(int(0.1 * length), int(0.9 * length))

        if disc_type == "step" or disc_type == "mixed":
            # Heaviside-Sprung
            height = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
            signal[pos:] += height

        if disc_type == "slope_change" or disc_type == "mixed":
            # Plötzliche Steigungsänderung
            new_slope = rng.uniform(-4.0, 4.0)
            signal[pos:] += new_slope * t[:length - pos]

    return signal


def generate_trend(
    trend_type: TrendType,
    length: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Erzeugt einen sauberen Trend der angegebenen Art.

    Parameters
    ----------
    trend_type : str
        Einer von "slow_trend", "periodic", "multi_periodic", "discontinuous".
    length : int
        Länge der Zeitreihe (default: 256).
    rng : np.random.Generator, optional
        Zufallsgenerator (wird neu erstellt wenn None).

    Returns
    -------
    np.ndarray  shape (length,), normalisiert auf [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    generators = {
        "slow_trend":    generate_slow_trend,
        "periodic":      generate_periodic,
        "multi_periodic": generate_multi_periodic,
        "discontinuous": generate_discontinuous,
    }

    if trend_type not in generators:
        raise ValueError(
            f"Unbekannter Trend-Typ '{trend_type}'. "
            f"Wähle aus: {list(generators.keys())}"
        )

    raw = generators[trend_type](length, rng)
    return normalize(raw)


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalisiert ein Array auf [-1, 1] (Min-Max-Skalierung)."""
    x_min, x_max = x.min(), x.max()
    if np.isclose(x_min, x_max):
        return np.zeros_like(x)
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

ALL_TREND_TYPES: list[TrendType] = [
    "slow_trend", "periodic", "multi_periodic", "discontinuous"
]


class TrendDataset(Dataset):
    """
    PyTorch Dataset, das synthetische Trend-Zeitreihen erzeugt.

    Jedes Item ist ein Dict mit:
        'x_clean'   : sauberer Trend  (L,)  float32
        'trend_type': Name des Trend-Typs (str)

    Das Hinzufügen von Rauschen übernimmt der NoiseScheduler im Training.

    Parameters
    ----------
    n_samples   : Anzahl der Datenpunkte im Dataset.
    seq_len     : Länge jeder Zeitreihe.
    trend_types : Welche Trend-Typen verwendet werden sollen.
    seed        : Reproduzierbarer Zufallsseed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seq_len: int = 256,
        trend_types: Optional[list[TrendType]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples   = n_samples
        self.seq_len     = seq_len
        self.trend_types = trend_types or ALL_TREND_TYPES
        self.rng         = np.random.default_rng(seed)

        # Alle Daten im Voraus generieren
        self._data = self._pregenerate()

    def _pregenerate(self) -> list[dict]:
        data = []
        for i in range(self.n_samples):
            trend_type = self.trend_types[i % len(self.trend_types)]
            x_clean = generate_trend(trend_type, self.seq_len, self.rng)
            data.append({"x_clean": x_clean.astype(np.float32), "trend_type": trend_type})
        return data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]
        return {
            "x_clean":    torch.from_numpy(item["x_clean"]),   # (L,)
            "trend_type": item["trend_type"],
        }


# ---------------------------------------------------------------------------
# Schnelltest / Vorschau
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    fig.suptitle("Synthetische Trends (je 4 Beispiele pro Typ)", fontsize=14)

    for col, ttype in enumerate(ALL_TREND_TYPES):
        axes[0, col].set_title(ttype, fontweight="bold")
        for row in range(4):
            trend = generate_trend(ttype, length=256, rng=rng)
            axes[row, col].plot(trend, linewidth=1.2)
            axes[row, col].set_ylim(-1.2, 1.2)
            axes[row, col].set_xlabel("Zeitschritt")
            if col == 0:
                axes[row, col].set_ylabel(f"Beispiel {row + 1}")

    plt.tight_layout()
    plt.savefig("trend_examples.png", dpi=150)
    plt.show()
    print("Plot gespeichert als 'trend_examples.png'.")
