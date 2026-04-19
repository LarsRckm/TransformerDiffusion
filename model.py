"""
model.py
========
Transformer-basierte Architektur für Diffusion-Modelle (Trend-Extraktion).

Aufbau:
  1. SinusoidalTimeEmbedding  – Kodiert Diffusionsschritt t als Vektor
  2. TimeConditionedMLP       – Verarbeitet Time-Embedding → (scale, shift)
  3. FiLMLayer                – Feature-wise Linear Modulation (scale*x + shift)
  4. PositionalEncoding       – Sinus/Cosinus-Positionskodierung für Zeitreihe
  5. TransformerBlock         – MHA + FiLM + FFN + LayerNorm (bidirektional)
  6. DiffusionTransformer     – Komplettes Modell (Feature-Proj → N Blocks → Output)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Kodiert den skalaren Diffusionsschritt t ∈ {0, …, T-1} in einen Vektor
    der Dimension `dim` mittels Sin/Cos-Encodings (wie im DDPM-Paper).

    Formel:
        emb[2k]   = sin(t / 10000^(2k/dim))
        emb[2k+1] = cos(t / 10000^(2k/dim))
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim muss gerade sein."
        self.dim = dim
        half = dim // 2
        # Fixe Frequenzen (kein Training)
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)   # (dim//2,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) long – Schritt-Indizes.

        Returns
        -------
        emb : (B, dim) float – Time-Embedding.
        """
        t = t.float().unsqueeze(1)          # (B, 1)
        args = t * self.freqs.unsqueeze(0)  # (B, dim//2)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# 2. Time-Conditioned MLP  (verarbeitet Time-Embedding → scale + shift)
# ---------------------------------------------------------------------------

class TimeConditionedMLP(nn.Module):
    """
    Kleines MLP: time_emb_dim → d_model (scale) + d_model (shift).

    FiLM-Konditionierung: out = scale * x + shift
    Ref: "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al.)
    """

    def __init__(self, time_emb_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * d_model),   # Ausgabe: [scale | shift]
        )

    def forward(self, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        time_emb : (B, time_emb_dim)

        Returns
        -------
        scale : (B, 1, d_model)
        shift : (B, 1, d_model)
        """
        out   = self.mlp(time_emb)                     # (B, 2 * d_model)
        scale, shift = out.chunk(2, dim=-1)             # je (B, d_model)
        return scale.unsqueeze(1), shift.unsqueeze(1)   # (B, 1, d_model)


# ---------------------------------------------------------------------------
# 3. FiLM Layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation:  out = (1 + scale) * x + shift
    Das (+1) beim Scale sorgt für ein stabiles residuales Signal.
    """

    def forward(
        self,
        x: torch.Tensor,        # (B, L, d_model)
        scale: torch.Tensor,    # (B, 1, d_model)
        shift: torch.Tensor,    # (B, 1, d_model)
    ) -> torch.Tensor:
        return (1.0 + scale) * x + shift


# ---------------------------------------------------------------------------
# 4. Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Standard Sinus/Cosinus-Positionskodierung über die Sequenzlänge.
    Bidirektional – kein kausales Masking.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, d_model, 2, dtype=torch.float)
            / d_model
        )    # (d_model//2,)

        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_model)

        Returns
        -------
        x + pe[:, :L, :] mit Dropout.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 5. Transformer Block mit FiLM Time-Conditioning
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Bidirektionaler Transformer-Block mit FiLM-Time-Conditioning.

    Aufbau pro Block:
        x  →  LayerNorm  →  Multi-Head Attention  →  FiLM(t)  →  Residual
           →  LayerNorm  →  FFN (MLP)              →  FiLM(t)  →  Residual
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-Head Self-Attention (KEIN causal masking)
        self.attn      = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(d_model)

        # Feedforward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn  = nn.LayerNorm(d_model)

        # FiLM-Konditionierung: ein MLP pro Block (Attention und FFN)
        self.film_attn = TimeConditionedMLP(time_emb_dim, d_model)
        self.film_ffn  = TimeConditionedMLP(time_emb_dim, d_model)
        self.film      = FiLMLayer()

    def forward(
        self,
        x: torch.Tensor,           # (B, L, d_model)
        time_emb: torch.Tensor,    # (B, time_emb_dim)
    ) -> torch.Tensor:

        # --- Attention ---
        scale, shift = self.film_attn(time_emb)          # je (B, 1, d_model)
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # bidirektional
        x = x + self.film(attn_out, scale, shift)

        # --- FFN ---
        scale, shift = self.film_ffn(time_emb)
        x_norm = self.norm_ffn(x)
        x = x + self.film(self.ffn(x_norm), scale, shift)

        return x


# ---------------------------------------------------------------------------
# 6. DiffusionTransformer – Komplettes Modell
# ---------------------------------------------------------------------------

class DiffusionTransformer(nn.Module):
    """
    Transformer-Modell fuer Diffusion-Denoising.

    Das Modell kann konditioniert betrieben werden: neben x_t und t kann eine
    beobachtete Zeitreihe y_obs (verrauscht und ggf. maskiert) sowie eine Maske
    mask (1=beobachtet, 0=fehlend) uebergeben werden. Optional kann zudem
    log(sigma) als Konditionierung eingespeist werden.

    Aufbau:
        (B, L, 1)  →  Feature Projection  →  (B, L, d_model)
                   →  + Positional Encoding
                   →  N × TransformerBlock(t)
                   →  Output Head
                   →  (B, L, 1)

    Parameters
    ----------
    seq_len         : Länge der Zeitreihe (Kontextfenster).
    d_model         : Modell-Dimension (Embedding-Größe).
    nhead           : Anzahl Attention-Heads (muss d_model teilen).
    num_layers      : Anzahl Transformer-Blöcke.
    dim_feedforward : Größe des FFN-Hidden-Layers.
    time_emb_dim    : Dimension des Time-Embeddings.
    dropout         : Dropout-Rate.
    """

    def __init__(
        self,
        seq_len: int = 256,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        time_emb_dim: int = 128,
        dropout: float = 0.1,
        in_channels: int = 3,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.d_model   = d_model

        # --- Time Embedding ---
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        # Projektion auf größere Dim. für expressiveres Embedding
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # --- Feature Projection: in_channels → d_model ---
        # Standard: [x_t, y_obs, mask]
        self.in_channels = in_channels
        self.input_proj = nn.Linear(in_channels, d_model)

        # --- Sigma-Conditioning (optional) ---
        # log(sigma) wird als zusaetzliche Konditionierung in das Time-Embedding addiert.
        self.sigma_proj = nn.Sequential(
            nn.Linear(1, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # --- Positional Encoding ---
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 64, dropout=dropout)

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # --- Output Head: d_model → 1 ---
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)

        # Gewichtsinitialisierung
        self._init_weights()

    def _init_weights(self):
        """Initialisiert Gewichte mit Xavier-Uniform für Linearschichten."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_t: torch.Tensor,     # (B, L, 1)  verrauschte Zeitreihe
        t: torch.Tensor,        # (B,)       Diffusionsschritt
        y_obs: Optional[torch.Tensor] = None,   # (B, L, 1)
        mask: Optional[torch.Tensor] = None,    # (B, L, 1)  1=observed, 0=missing
        sigma_log: Optional[torch.Tensor] = None,  # (B,) log(sigma) in model space
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, L, 1) – verrauschte Zeitreihe.
        t   : (B,) long  – Diffusionsschritt je Batch-Element.

        Returns
        -------
        eps_hat : (B, L, 1) – Vorhergesagtes Rauschen ε̂.
        """
        # 1. Time Embedding (+ optional sigma conditioning)
        t_emb = self.time_embedding(t)    # (B, time_emb_dim)
        t_emb = self.time_proj(t_emb)     # (B, time_emb_dim)
        if sigma_log is not None:
            t_emb = t_emb + self.sigma_proj(sigma_log.float().unsqueeze(-1))

        # 2. Conditioning Defaults (Legacy-Aufrufe)
        if y_obs is None:
            y_obs = torch.zeros_like(x_t)
        if mask is None:
            mask = torch.ones_like(x_t)

        # 3. Feature Projection + Positional Encoding
        inp = torch.cat([x_t, y_obs, mask], dim=-1)  # (B, L, in_channels)
        x = self.input_proj(inp)          # (B, L, d_model)
        x = self.pos_enc(x)               # (B, L, d_model)

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, t_emb)           # (B, L, d_model)

        # 4. Output Head
        x = self.output_norm(x)
        eps_hat = self.output_head(x)     # (B, L, 1)

        return eps_hat

    def count_parameters(self) -> int:
        """Gibt die Anzahl trainierbarer Parameter zurück."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SigmaEstimator(nn.Module):
    """Schätzt log(sigma) aus (y_obs, mask) pro Sample.

    Input:
        y_obs : (B, L, 1)
        mask  : (B, L, 1)
    Output:
        log_sigma_hat : (B,)
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y_obs, mask], dim=-1)  # (B, L, 2)
        x = x.transpose(1, 2)                 # (B, 2, L)
        h = self.net(x)                       # (B, hidden, L)
        h = h.mean(dim=-1)                    # (B, hidden)
        return self.head(h).squeeze(-1)       # (B,)


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== DiffusionTransformer Schnelltest ===\n")

    model = DiffusionTransformer(
        seq_len=256,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        time_emb_dim=128,
        dropout=0.1,
    )

    n_params = model.count_parameters()
    print(f"Trainierbare Parameter: {n_params:,}")

    # Dummy-Daten
    B, L = 4, 256
    x_t = torch.randn(B, L, 1)
    t   = torch.randint(0, 1000, (B,))

    # Forward Pass
    with torch.no_grad():
        eps_hat = model(x_t, t)

    print(f"Input shape:  {x_t.shape}")
    print(f"t shape:      {t.shape}")
    print(f"Output shape: {eps_hat.shape}")
    assert eps_hat.shape == (B, L, 1), "Output-Shape stimmt nicht!"

    print("\n✓ Forward Pass erfolgreich.")
