"""
data_generator.py
=================
Synthetischer Daten-Generator für Trendextraktion mit Diffusion-Modellen.

Trend-Typen (Basis):
  - slow_trend:           Langsam ändernde Trends (Polynom niedriger Ordnung ODER
                          iterativer Random-Walk + Spline-Glättung)
  - periodic:             Einzelne Sinus/Cosinus-Welle
  - multi_periodic:       Überlagerung mehrerer Sinuswellen (Fourier-Stil)
  - discontinuous:        Trends mit Sprüngen oder plötzlichen Steigungsänderungen

Erweiterte Trend-Typen:
  - exponential_decay:    Exponentiell ab- oder ansteigende Kurve (mit/ohne Offset)
  - chirp:                Sinuswelle mit linear ansteigender Frequenz (Frequenz-Sweep)
  - damped_oscillation:   Gedämpfte Sinusschwingung (exponentieller Envelope)
  - logistic_growth:      S-förmige Wachstumskurve (Logistisch / Sigmoid)
  - sawtooth_wave:        Sägezahn-artiger periodischer Trend
  - random_walk_trend:    Gefilterte Brownsche Bewegung (glatter als slow_trend walk)

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

TrendType = Literal[
    "slow_trend", "periodic", "multi_periodic", "discontinuous",
    "exponential_decay", "chirp", "damped_oscillation",
    "logistic_growth", "sawtooth_wave", "random_walk_trend",
]


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


# ---------------------------------------------------------------------------
# Erweiterte Trend-Generatoren
# ---------------------------------------------------------------------------

def generate_exponential_decay(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Exponentiell ab- oder ansteigende Kurve, gelegentlich mit überlagerten
    Sinusschwingungen oder einem konstanten Offset.

    Varianten (zufällig gewählt):
      - "decay"   : Exponentieller Abfall,  f(t) = A · exp(-λ·t) + c
      - "growth"  : Exponentielles Wachstum, f(t) = A · exp(+λ·t) + c
      - "mixed"   : Abfall + leichte Überlagerung einer langsamen Schwingung
    """
    t = np.linspace(0.0, 1.0, length)
    variant = rng.choice(["decay", "growth", "mixed"])
    lam = rng.uniform(2.0, 8.0)          # Abklingkonstante
    A   = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
    c   = rng.uniform(-0.5, 0.5)        # Offset

    if variant == "decay":
        signal = A * np.exp(-lam * t) + c
    elif variant == "growth":
        signal = A * np.exp(lam * (t - 1.0)) + c   # von links: klein → groß
    else:  # mixed
        signal = A * np.exp(-lam * t) + c
        freq   = rng.uniform(1.0, 3.0)
        signal += 0.3 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return signal


def generate_chirp(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Frequenz-Sweep (Chirp): Sinuswelle, deren Frequenz linear von f_start
    nach f_end ansteigt (oder abfällt). Erzeugt ein zeitlich variierendes
    Muster, das sich fundamental von stationären Schwingungen unterscheidet.

    f(t) = A · sin(2π · (f_start·t + 0.5·(f_end - f_start)·t²) + φ)
    """
    t      = np.linspace(0.0, 1.0, length)
    f_start = rng.uniform(0.5, 3.0)
    f_end   = rng.uniform(5.0, 20.0)
    if rng.random() > 0.5:               # Manchmal von hoch nach niedrig
        f_start, f_end = f_end, f_start
    A     = rng.uniform(0.6, 1.5)
    phase = rng.uniform(0.0, 2 * np.pi)
    inst_phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t ** 2)
    return A * np.sin(inst_phase + phase)


def generate_damped_oscillation(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Gedämpfte Sinusschwingung:
      f(t) = A · exp(-γ·t) · sin(2π·f·t + φ)

    Der exponentielle Envelope sorgt dafür, dass die Amplitude mit der Zeit
    abnimmt – ein Muster, das in vielen physikalischen Systemen auftritt.
    Gelegentlich wird der Onset verzögert (Signal startet mit einer Rampe).
    """
    t = np.linspace(0.0, 1.0, length)
    A     = rng.uniform(0.8, 2.0)
    gamma = rng.uniform(2.0, 8.0)   # Dämpfungskonstante
    freq  = rng.uniform(2.0, 10.0)
    phase = rng.uniform(0.0, 2 * np.pi)

    envelope = np.exp(-gamma * t)
    signal   = A * envelope * np.sin(2 * np.pi * freq * t + phase)

    # Optionaler positiver Onset-Offset
    if rng.random() > 0.5:
        signal += rng.uniform(-0.5, 0.5)
    return signal


def generate_logistic_growth(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    S-förmige Wachstumskurve (Logistisch / Sigmoid):
      f(t) = L / (1 + exp(-k·(t - t0))) + c

    Parameter:
      L  : Sättigungswert
      k  : Wachstumsrate
      t0 : Wendepunkt (Mitte des Übergangs)
      c  : Basisoffset

    Manchmal gespiegelt (abfallende Kurve) oder mit einer kleinen
    überlagerten Schwingung kombiniert.
    """
    t  = np.linspace(-6.0, 6.0, length)
    L  = rng.uniform(1.5, 3.0)
    k  = rng.uniform(0.6, 2.5)
    t0 = rng.uniform(-2.0, 2.0)
    c  = rng.uniform(-0.3, 0.3)

    signal = L / (1.0 + np.exp(-k * (t - t0))) + c
    if rng.random() > 0.5:
        signal = -signal + signal.mean() * 2   # spiegeln
    return signal


def generate_sawtooth_wave(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sägezahn-artiger periodischer Trend. Drei Varianten:
      - "sawtooth"  : Klassisch ansteigender Sägezahn (Modulo-Funktion)
      - "triangle"  : Dreieckswelle (aufwärts + abwärts)
      - "asymmetric": Asymmetrischer Sägezahn mit unterschiedlicher
                      Anstiegs- und Abfallgeschwindigkeit
    """
    t = np.linspace(0.0, 1.0, length)
    variant = rng.choice(["sawtooth", "triangle", "asymmetric"])
    freq    = rng.uniform(1.5, 6.0)   # Anzahl Perioden
    amp     = rng.uniform(0.7, 1.5)

    phase = t * freq
    if variant == "sawtooth":
        signal = amp * 2.0 * (phase - np.floor(phase + 0.5))
    elif variant == "triangle":
        p = phase % 1.0
        signal = amp * (2.0 * np.where(p < 0.5, p, 1.0 - p) * 2.0 - 1.0)
    else:  # asymmetric
        duty = rng.uniform(0.2, 0.8)
        p = phase % 1.0
        ascending  = p / duty
        descending = (1.0 - p) / (1.0 - duty)
        signal = amp * np.where(p < duty, ascending, 1.0 - descending) * 2.0 - amp
    return signal


def generate_random_walk_trend(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Gefilterte Brownsche Bewegung als glatter Trend. Im Gegensatz zu
    ``generate_slow_trend`` wird hier ausschließlich ein Random Walk
    erzeugt, der durch einen Gauss-Filter stark geglättet wird.
    Die Tiefpass-Bandbreite wird zufällig gewählt, sodass unterschiedlich
    "träge" Trends entstehen.
    """
    from scipy.ndimage import gaussian_filter1d

    steps  = rng.standard_normal(length)
    signal = np.cumsum(steps)

    # Sigma in Zeitschritten: groß → sehr glatt, klein → leicht rauer
    sigma = rng.uniform(length * 0.03, length * 0.2)
    signal = gaussian_filter1d(signal, sigma=sigma)
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
        Einer von: slow_trend, periodic, multi_periodic, discontinuous,
        exponential_decay, chirp, damped_oscillation, logistic_growth,
        sawtooth_wave, random_walk_trend.
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
        "slow_trend":         generate_slow_trend,
        "periodic":           generate_periodic,
        "multi_periodic":     generate_multi_periodic,
        "discontinuous":      generate_discontinuous,
        # --- Erweiterte Typen ---
        "exponential_decay":  generate_exponential_decay,
        "chirp":              generate_chirp,
        "damped_oscillation": generate_damped_oscillation,
        "logistic_growth":    generate_logistic_growth,
        "sawtooth_wave":      generate_sawtooth_wave,
        "random_walk_trend":  generate_random_walk_trend,
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

# Trainings-Typen (werden vom Modell im Training gesehen)
TRAIN_TREND_TYPES: list[TrendType] = [
    "slow_trend", "periodic", "multi_periodic", "discontinuous"
]

# Alle bekannten Typen (inkl. erweiterter Generatoren)
ALL_TREND_TYPES: list[TrendType] = [
    "slow_trend", "periodic", "multi_periodic", "discontinuous",
    "exponential_decay", "chirp", "damped_oscillation",
    "logistic_growth", "sawtooth_wave", "random_walk_trend",
]

# Nur die erweiterten (neuen) Typen
EXTENDED_TREND_TYPES: list[TrendType] = [
    "exponential_decay", "chirp", "damped_oscillation",
    "logistic_growth", "sawtooth_wave", "random_walk_trend",
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


class OnTheFlyTrendDataset(Dataset):
    """Lazy/on-the-fly Dataset.

    Generates a fresh synthetic clean trend in __getitem__ without precomputing
    the full dataset in memory.

    Note: For reproducibility and stable validation splits, the sample for a
    given idx is deterministic (seed + idx). Diversity comes from large
    n_samples and randomization inside the generators.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seq_len: int = 256,
        trend_types: Optional[list[TrendType]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.trend_types = trend_types or ALL_TREND_TYPES
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + int(idx))
        trend_type = rng.choice(self.trend_types)
        # Randomize within each type
        x_clean = generate_trend(trend_type, self.seq_len, rng)
        return {
            "x_clean": torch.from_numpy(x_clean.astype(np.float32)),
            "trend_type": trend_type,
        }


# ---------------------------------------------------------------------------
# Schnelltest / Vorschau
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n_cols = len(ALL_TREND_TYPES)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.5 * n_cols, 12))
    fig.suptitle("Synthetische Trends (je 4 Beispiele pro Typ)", fontsize=14)

    for col, ttype in enumerate(ALL_TREND_TYPES):
        axes[0, col].set_title(ttype, fontweight="bold", fontsize=8)
        for row in range(4):
            trend = generate_trend(ttype, length=256, rng=rng)
            axes[row, col].plot(trend, linewidth=1.2)
            axes[row, col].set_ylim(-1.2, 1.2)
            axes[row, col].set_xlabel("Zeitschritt", fontsize=7)
            if col == 0:
                axes[row, col].set_ylabel(f"Beispiel {row + 1}")

    plt.tight_layout()
    plt.savefig("trend_examples.png", dpi=150)
    plt.show()
    print("Plot gespeichert als 'trend_examples.png'.")
