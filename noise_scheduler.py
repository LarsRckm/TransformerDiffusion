"""
noise_scheduler.py
==================
Forward-Prozess und Beta-Schedule für das Diffusion-Modell.

Implementiert:
  - Linear Beta-Schedule
  - Cosine Beta-Schedule  (Nichol & Dhariwal, 2021)
  - q_sample: Fügt Rauschen zu x_0 hinzu für einen beliebigen Schritt t
  - Normalverteiltes und Laplace-verteiltes Rauschen
  - p_losses: Berechnet den Training-Loss (Huber oder MSE)
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NoiseType    = Literal["gaussian", "laplace", "mixed"]
ScheduleType = Literal["linear", "cosine"]


# ---------------------------------------------------------------------------
# Beta-Schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Gleichmäßig steigender Beta-Plan von beta_start bis beta_end.
    Klassischer DDPM-Schedule (Ho et al., 2020).
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine-Schedule (Nichol & Dhariwal, 2021).
    Sanfterer Abfall am Anfang und Ende – gut für Zeitreihen.

        alpha_bar(t) = cos((t/T + s) / (1 + s) * pi/2)^2
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]              # normalisieren auf 1 bei t=0
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, min=1e-5, max=0.9999).float()


# ---------------------------------------------------------------------------
# Rauschen samplen
# ---------------------------------------------------------------------------

def sample_noise(shape: tuple, noise_type: NoiseType, device: torch.device) -> torch.Tensor:
    """
    Samplet Rauschen der angegebenen Art.

    Parameters
    ----------
    shape      : Ausgabe-Shape (z.B. (B, L, 1)).
    noise_type : "gaussian", "laplace" oder "mixed" (50/50 per Sample).
    device     : torch.device.

    Returns
    -------
    torch.Tensor mit Rauschen, selbe Shape wie `shape`.
    """
    if noise_type == "gaussian":
        return torch.randn(shape, device=device)

    elif noise_type == "laplace":
        # Laplace(0, 1/sqrt(2)) hat Varianz 1  (wie N(0,1))
        u = torch.rand(shape, device=device) - 0.5
        return -(1.0 / math.sqrt(2)) * u.sign() * torch.log1p(-2 * u.abs())

    elif noise_type == "mixed":
        # Batch-weise zufällig Gauß oder Laplace
        B = shape[0]
        gaussian = torch.randn(shape, device=device)
        laplace_u = torch.rand(shape, device=device) - 0.5
        laplace = -(1.0 / math.sqrt(2)) * laplace_u.sign() * torch.log1p(-2 * laplace_u.abs())
        # Maske: 50 % Gauß, 50 % Laplace pro Batch-Element
        mask = (torch.rand(B, *([1] * (len(shape) - 1)), device=device) > 0.5)
        return torch.where(mask, gaussian, laplace)

    else:
        raise ValueError(f"Unbekannter noise_type '{noise_type}'. Wähle 'gaussian', 'laplace' oder 'mixed'.")


# ---------------------------------------------------------------------------
# NoiseScheduler
# ---------------------------------------------------------------------------

class NoiseScheduler(nn.Module):
    """
    Verwaltet den Forward-Prozess q(x_t | x_0) für DDPM.

    Formel (reparametrisiert):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Parameters
    ----------
    T             : Anzahl der Diffusionsschritte (default: 1000).
    schedule_type : "linear" oder "cosine".
    beta_start    : Startwert für den Beta-Schedule (nur Linear).
    beta_end      : Endwert für den Beta-Schedule (nur Linear).
    """

    def __init__(
        self,
        T: int = 1000,
        schedule_type: ScheduleType = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.T = T
        self.schedule_type = schedule_type

        # Beta-Schedule
        if schedule_type == "linear":
            betas = linear_beta_schedule(T, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"Unbekannter schedule_type '{schedule_type}'.")

        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Buffer registrieren (werden mit .to(device) mitbewegt)
        self.register_buffer("betas",               betas)
        self.register_buffer("alphas",              alphas)
        self.register_buffer("alphas_cumprod",      alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Ableitungen für q_sample
        self.register_buffer("sqrt_alphas_cumprod",       torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

        # Ableitungen für p_sample (Reversal)
        self.register_buffer("sqrt_recip_alphas",    torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance",
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Hilfsfunktion: Wert bei Index t extrahieren, passend zur Batch-Dim.
    # ------------------------------------------------------------------
    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extrahiert a[t] für jeden Batch-Eintrag und bringt es auf die Form
        (B, 1, 1, ...) passend zu x_shape.
        """
        B = t.shape[0]
        out = a.gather(0, t)          # (B,)
        return out.reshape(B, *((1,) * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    # Forward Process: q(x_t | x_0)
    # ------------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        noise_type: NoiseType = "mixed",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Erzeugt x_t aus x_0 durch direktes Rauschen-Hinzufügen.

        Parameters
        ----------
        x_0        : Saubere Zeitreihe (B, L, 1).
        t          : Diffusionsschritt je Batch-Element (B,), long.
        noise      : Vorbereitetes Rauschen (optional, wird sonst gesampelt).
        noise_type : Art des Rauschens.

        Returns
        -------
        (x_t, noise)  – verrauschte Zeitreihe und das verwendete Rauschen.
        """
        if noise is None:
            noise = sample_noise(x_0.shape, noise_type, x_0.device)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Training-Loss: L_simple
    # ------------------------------------------------------------------
    def p_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise_type: NoiseType = "mixed",
        loss_type: Literal["mse", "huber"] = "huber",
    ) -> torch.Tensor:
        """
        Berechnet den Diffusion-Trainingsloss.

        Steps:
          1. Noise samplen
          2. x_t = q_sample(x_0, t, noise)
          3. Modell sagt Noise vorher: eps_hat = model(x_t, t)
          4. Loss zwischen eps und eps_hat

        Parameters
        ----------
        model      : DiffusionTransformer Instanz.
        x_0        : Saubere Zeitreihe (B, L, 1).
        t          : Schritt-Indizes (B,).
        noise_type : "gaussian", "laplace" oder "mixed".
        loss_type  : "mse" (L2) oder "huber" (robust gegen Ausreißer).

        Returns
        -------
        Skalar-Loss.
        """
        x_t, noise = self.q_sample(x_0, t, noise_type=noise_type)

        # Modell sagt Rauschen vorher
        eps_hat = model(x_t, t)       # (B, L, 1)

        if loss_type == "mse":
            return F.mse_loss(eps_hat, noise)
        elif loss_type == "huber":
            return F.huber_loss(eps_hat, noise, delta=1.0)
        else:
            raise ValueError(f"Unbekannter loss_type '{loss_type}'.")

    # ------------------------------------------------------------------
    # Reverse Process: ein Schritt p(x_{t-1} | x_t)  (DDPM)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Ein reversiver DDPM-Schritt: x_t → x_{t-1}.

        Parameters
        ----------
        model         : Trainiertes DiffusionTransformer-Modell.
        x_t           : Aktuell verrauschte Zeitreihe (B, L, 1).
        t             : Aktueller Schritt (int, gleich für ganzen Batch).
        clip_denoised : Ob x_0-Schätzung auf [-1, 1] geclippt wird.

        Returns
        -------
        x_{t-1}  (B, L, 1).
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        eps_hat      = model(x_t, t_tensor)
        sqrt_recip_a = self._extract(self.sqrt_recip_alphas,    t_tensor, x_t.shape)
        beta_t       = self._extract(self.betas,                 t_tensor, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)

        # Schätzung von x_{t-1} (Gleichung 11 aus Ho et al., 2020)
        model_mean = sqrt_recip_a * (x_t - beta_t / sqrt_one_minus * eps_hat)

        if t == 0:
            return model_mean
        else:
            posterior_var = self._extract(self.posterior_variance, t_tensor, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var) * noise

    # ------------------------------------------------------------------
    # Reverse Process: DDPM vollständig (T → 0)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        x_noisy: torch.Tensor,
        start_t: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Vollständiger DDPM Reverse-Prozess (T Schritte).

        Parameters
        ----------
        model    : Trainiertes Modell.
        x_noisy  : Verrauschte Zeitreihe als Startpunkt (B, L, 1).
                   Wenn start_t=None, startet bei T-1 (komplett verrauscht).
        start_t  : Welches t als Startpunkt verwendet wird.

        Returns
        -------
        x_0: Extrahierter Trend (B, L, 1).
        """
        model.eval()
        x = x_noisy.clone()
        start = self.T - 1 if start_t is None else start_t

        for t in reversed(range(0, start + 1)):
            x = self.p_sample(model, x, t)

        return x

    # ------------------------------------------------------------------
    # Reverse Process: DDIM (schnell, deterministisch)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        x_noisy: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        start_t: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Schneller DDIM Reverse-Prozess (Song et al., 2020).

        Parameters
        ----------
        model      : Trainiertes Modell.
        x_noisy    : Startpunkt (B, L, 1).
        ddim_steps : Anzahl der Schritte (<<T möglich).
        eta        : Stochastizität (0 = deterministisch, 1 = wie DDPM).
        start_t    : Start-Schritt (default: T-1).

        Returns
        -------
        x_0: Extrahierter Trend (B, L, 1).
        """
        model.eval()
        B = x_noisy.shape[0]
        start = self.T - 1 if start_t is None else start_t

        # Gleichmäßig verteilte Zeitschritte
        timesteps = torch.linspace(start, 0, ddim_steps + 1).long().tolist()

        x = x_noisy.clone()

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]

            t_tensor = torch.full((B,), t_curr, device=x.device, dtype=torch.long)

            # Koeffizienten
            a_t   = self.alphas_cumprod[t_curr]
            a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

            eps_hat = model(x, t_tensor)

            # x_0 schätzen
            x0_pred = (x - torch.sqrt(1 - a_t) * eps_hat) / torch.sqrt(a_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # Rauschen für diese Iteration
            sigma = (
                eta
                * torch.sqrt((1 - a_prev) / (1 - a_t))
                * torch.sqrt(1 - a_t / a_prev)
            )
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev - sigma**2) * eps_hat + sigma * noise

        return x


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== NoiseScheduler Schnelltest ===")

    scheduler = NoiseScheduler(T=1000, schedule_type="cosine")
    print(f"Beta[0]   = {scheduler.betas[0]:.6f}")
    print(f"Beta[999] = {scheduler.betas[999]:.6f}")
    print(f"AlphaBar[0]   = {scheduler.alphas_cumprod[0]:.6f}")
    print(f"AlphaBar[999] = {scheduler.alphas_cumprod[999]:.6f}")

    # Test q_sample
    B, L = 4, 256
    x0 = torch.randn(B, L, 1)
    t  = torch.randint(0, 1000, (B,))
    x_t, noise = scheduler.q_sample(x0, t, noise_type="mixed")
    print(f"\nq_sample: x0 shape={x0.shape}, x_t shape={x_t.shape}, noise shape={noise.shape}")

    # Visualisierung: Beta-Schedule
    import matplotlib.pyplot as plt
    lin = NoiseScheduler(T=1000, schedule_type="linear")
    cos = NoiseScheduler(T=1000, schedule_type="cosine")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(lin.betas.numpy(), label="linear")
    axes[0].plot(cos.betas.numpy(), label="cosine")
    axes[0].set_title("Beta-Schedule")
    axes[0].legend()

    axes[1].plot(lin.alphas_cumprod.numpy(), label="linear")
    axes[1].plot(cos.alphas_cumprod.numpy(), label="cosine")
    axes[1].set_title("Kumulatives Alpha (Signalstärke)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("beta_schedule.png", dpi=150)
    plt.show()
    print("Plot gespeichert als 'beta_schedule.png'.")
