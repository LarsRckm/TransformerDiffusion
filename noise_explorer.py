"""
noise_explorer.py
=================
Interaktives Tool zum Erkunden des Forward-Diffusion-Prozesses.

Zeigt, wie sich eine Zeitreihe bei Schritt t verändert:
    x_t = sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε

Steuerung:
  - Slider "t"        : Diffusionsschritt (0 = kein Rauschen, T-1 = max Rauschen)
  - Slider "T"        : Gesamtanzahl Diffusionsschritte
  - Radio "Schedule"  : linear oder cosine
  - Radio "Trendtyp"  : periodic / slow_trend / multi_periodic / discontinuous
  - Slider "Seed"     : andere Rausch-Realisierung

Aufruf:
    python noise_explorer.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ---------------------------------------------------------------------------
# Schedule-Berechnung
# ---------------------------------------------------------------------------

def linear_alpha_bar(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    return np.cumprod(1 - betas)

def cosine_alpha_bar(T, s=0.008):
    steps = np.arange(T + 1)
    ab = np.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = 1.0 - ab[1:] / ab[:-1]
    betas = np.clip(betas, 1e-5, 0.9999)
    return np.cumprod(1 - betas)

def get_alpha_bars(T, schedule):
    if schedule == "cosine":
        return cosine_alpha_bar(T)
    else:
        return linear_alpha_bar(T)

# ---------------------------------------------------------------------------
# Trendgeneratoren
# ---------------------------------------------------------------------------

def normalize(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if np.isclose(lo, hi) else 2*(x-lo)/(hi-lo)-1

def gen_slow_trend(L, rng):
    t = np.linspace(0, 1, L)
    d = rng.integers(1, 4)
    c = rng.uniform(-1, 1, size=d+1)
    return sum(c[k] * t**k for k in range(d+1))

def gen_periodic(L, rng):
    f = rng.uniform(1.5, 6.0)
    p = rng.uniform(0, 2*math.pi)
    return np.sin(f * np.linspace(0, 2*math.pi, L) + p)

def gen_multi_periodic(L, rng):
    t = np.linspace(0, 2*math.pi, L)
    y = np.zeros(L)
    for _ in range(rng.integers(2, 5)):
        y += rng.uniform(.2,1.) * np.sin(rng.uniform(1,8)*t + rng.uniform(0, 2*math.pi))
    return y

def gen_discontinuous(L, rng):
    t = np.linspace(0, 1, L)
    nb = rng.integers(2, 5)
    br = np.sort(rng.uniform(0.1, 0.9, nb))
    sl = rng.uniform(-3, 3, nb+1)
    of = np.zeros(nb+1)
    for i in range(1, nb+1):
        of[i] = of[i-1] + sl[i-1]*(br[i-1]-(br[i-2] if i>1 else 0))
    y = np.zeros(L)
    edges = np.concatenate([[0], br, [1]])
    for i in range(len(edges)-1):
        m = (t>=edges[i])&(t<edges[i+1])
        y[m] = sl[i]*(t[m]-edges[i])+of[i]
    return y

GENERATORS = {
    "periodic":      gen_periodic,
    "slow_trend":    gen_slow_trend,
    "multi_periodic":gen_multi_periodic,
    "discontinuous": gen_discontinuous,
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

L       = 1000
RNG     = np.random.default_rng(42)
T_INIT  = 1000
t_INIT  = 500

state = {
    "schedule":   "cosine",
    "trend":      "periodic",
    "T":          T_INIT,
    "t":          t_INIT,
    "seed":       0,
}

def make_x0():
    return normalize(GENERATORS[state["trend"]](L, RNG))

x0 = make_x0()

def get_noisy():
    ab   = get_alpha_bars(state["T"], state["schedule"])
    t    = min(state["t"], state["T"]-1)
    ab_t = ab[t]
    eps  = np.random.default_rng(state["seed"]).standard_normal(L)
    x_t  = math.sqrt(ab_t) * x0 + math.sqrt(1 - ab_t) * eps
    return x_t, ab_t, math.sqrt(ab_t), math.sqrt(1 - ab_t)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

BG      = "#1a1a2e"
PANEL   = "#16213e"
C_CLEAN = "#4fc3f7"
C_NOISY = "#ffb74d"
C_DIFF  = "#ef5350"
C_AB    = "#a5d6a7"
WHITE   = "white"

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("Forward-Diffusion Explorer  ·  x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε",
             color=WHITE, fontsize=12, fontweight="bold", y=0.97)

# Signal panels
gs = fig.add_gridspec(3, 1, left=0.06, right=0.62, top=0.91, bottom=0.07,
                      hspace=0.45)
ax_c = fig.add_subplot(gs[0])
ax_n = fig.add_subplot(gs[1])
ax_d = fig.add_subplot(gs[2])

for ax, title in [
    (ax_c, "Ground Truth  x₀"),
    (ax_n, "Verrauscht  x_t"),
    (ax_d, "Rausch-Differenz  (x_t − x₀)"),
]:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE, labelsize=7)
    ax.set_xlim(0, L)
    ax.grid(True, alpha=0.2, color="#555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

ta = np.arange(L)
x_t0, ab0, s_ab0, s_1mab0 = get_noisy()

ln_c,  = ax_c.plot(ta, x0,  color=C_CLEAN, lw=1.5)
ln_n,  = ax_n.plot(ta, x_t0, color=C_NOISY, lw=0.8, alpha=0.9)
ln_cr, = ax_n.plot(ta, x0,  color=C_CLEAN, lw=0.8, alpha=0.35)
ln_d,  = ax_d.plot(ta, x_t0-x0, color=C_DIFF, lw=0.7)
ax_d.axhline(0, color="#666", lw=0.5, ls="--")

for ax in [ax_c, ax_n]:
    ax.set_ylim(-2.2, 2.2)

# Schedule-Überblick (rechts oben)
ax_sch = fig.add_axes([0.66, 0.70, 0.30, 0.20], facecolor=PANEL)
ax_sch.set_facecolor(PANEL)
ax_sch.tick_params(colors=WHITE, labelsize=7)
ax_sch.set_title("Schedule: √ᾱ_t  /  √(1-ᾱ_t)", color=WHITE, fontsize=8)
ax_sch.set_xlabel("t", color=WHITE, fontsize=7)
ax_sch.set_xlim(0, T_INIT); ax_sch.set_ylim(0, 1.05)
ax_sch.grid(True, alpha=0.2, color="#555")
for sp in ax_sch.spines.values(): sp.set_edgecolor("#444")

ab_full = get_alpha_bars(T_INIT, "cosine")
ts_full = np.arange(T_INIT)

ln_sch_ab,  = ax_sch.plot(ts_full, np.sqrt(ab_full),   color=C_CLEAN, lw=1.2, label="√ᾱ")
ln_sch_1ab, = ax_sch.plot(ts_full, np.sqrt(1-ab_full), color=C_NOISY, lw=1.2, label="√(1-ᾱ)")
vline_sch   = ax_sch.axvline(t_INIT, color=WHITE, lw=1.0, ls="--", alpha=0.8)
ax_sch.legend(fontsize=7, labelcolor=WHITE, facecolor=PANEL, edgecolor="#444")

# Info-Box
ax_info = fig.add_axes([0.66, 0.48, 0.30, 0.20], facecolor=PANEL)
ax_info.set_xticks([]); ax_info.set_yticks([])
for sp in ax_info.spines.values(): sp.set_edgecolor("#444")
txt_info = ax_info.text(0.05, 0.93, "", color=WHITE, fontsize=9,
                        va="top", transform=ax_info.transAxes, family="monospace")

def update_info(ab_t, s_ab, s_1mab):
    t = state["t"]; T = state["T"]
    snr = ab_t / (1 - ab_t + 1e-9)
    snr_db = 10 * math.log10(snr + 1e-9)
    txt_info.set_text(
        f"t        = {t}  /  T = {T}\n"
        f"Schedule = {state['schedule']}\n"
        f"──────────────────\n"
        f"ᾱ_t      = {ab_t:.4f}\n"
        f"1 - ᾱ_t  = {1-ab_t:.4f}\n"
        f"√ᾱ_t     = {s_ab:.4f}   (Signal)\n"
        f"√(1-ᾱ_t) = {s_1mab:.4f}  (Rauschen)\n"
        f"──────────────────\n"
        f"SNR      = {snr:.3f}\n"
        f"SNR (dB) = {snr_db:.1f} dB\n"
        f"──────────────────\n"
        f"Trend    = {state['trend']}"
    )

update_info(ab0, s_ab0, s_1mab0)

# ---------------------------------------------------------------------------
# Widgets (rechts unten)
# ---------------------------------------------------------------------------

SCLR = "#2a2a4a"

def mk_slider(rect, label, vmin, vmax, vinit, valstep=None, color=C_CLEAN):
    ax = fig.add_axes(rect, facecolor=SCLR)
    kw = dict(valinit=vinit, color=color)
    if valstep: kw["valstep"] = valstep
    sl = Slider(ax, label, vmin, vmax, **kw)
    sl.label.set_color(WHITE); sl.valtext.set_color(WHITE)
    return sl

sl_t    = mk_slider([0.66, 0.38, 0.30, 0.025], "t",      0, T_INIT-1, t_INIT, valstep=1)
sl_T    = mk_slider([0.66, 0.32, 0.30, 0.025], "T",    100, 2000,     T_INIT, valstep=50,  color=C_AB)
sl_seed = mk_slider([0.66, 0.26, 0.30, 0.025], "Seed",   0, 99,       0,      valstep=1,   color="#ce93d8")

# Schedule radio
ax_r_sch = fig.add_axes([0.66, 0.13, 0.13, 0.10], facecolor=PANEL)
r_sch = RadioButtons(ax_r_sch, ["cosine", "linear"], active=0, activecolor=C_CLEAN)
for lb in r_sch.labels: lb.set_color(WHITE); lb.set_fontsize(8)
ax_r_sch.set_title("Schedule", color=WHITE, fontsize=8)

# Trend radio
ax_r_tr = fig.add_axes([0.81, 0.07, 0.17, 0.17], facecolor=PANEL)
r_tr = RadioButtons(ax_r_tr, list(GENERATORS.keys()), active=0, activecolor=C_NOISY)
for lb in r_tr.labels: lb.set_color(WHITE); lb.set_fontsize(8)
ax_r_tr.set_title("Trendtyp", color=WHITE, fontsize=8)

# ---------------------------------------------------------------------------
# Update-Logik
# ---------------------------------------------------------------------------

def redraw():
    x_t, ab_t, s_ab, s_1mab = get_noisy()

    ln_n.set_ydata(x_t)
    ln_cr.set_ydata(x0)
    diff = x_t - x0
    ln_d.set_ydata(diff)

    # Achsen
    ym = max(2.2, float(np.abs(x_t).max()) * 1.15)
    ax_n.set_ylim(-ym, ym)
    dm = max(0.05, float(np.abs(diff).max()) * 1.15)
    ax_d.set_ylim(-dm, dm)

    # Schedule-Plot neu zeichnen
    T = state["T"]
    ab_new = get_alpha_bars(T, state["schedule"])
    ts_new = np.arange(T)
    ln_sch_ab.set_data(ts_new,  np.sqrt(ab_new))
    ln_sch_1ab.set_data(ts_new, np.sqrt(1-ab_new))
    ax_sch.set_xlim(0, T)
    t_clamp = min(state["t"], T-1)
    vline_sch.set_xdata([t_clamp, t_clamp])

    update_info(ab_t, s_ab, s_1mab)
    fig.canvas.draw_idle()

def on_t(val):
    state["t"] = int(val)
    # t-Slider-Max anpassen falls T geändert wurde
    redraw()

def on_T(val):
    new_T = int(val)
    state["T"] = new_T
    # t darf T-1 nicht überschreiten
    if state["t"] >= new_T:
        state["t"] = new_T - 1
        sl_t.set_val(state["t"])
    sl_t.valmax = new_T - 1
    sl_t.ax.set_xlim(0, new_T - 1)
    redraw()

def on_seed(val):
    state["seed"] = int(val)
    redraw()

def on_schedule(label):
    state["schedule"] = label
    redraw()

def on_trend(label):
    global x0
    state["trend"] = label
    x0 = make_x0()
    ln_c.set_ydata(x0)
    redraw()

sl_t.on_changed(on_t)
sl_T.on_changed(on_T)
sl_seed.on_changed(on_seed)
r_sch.on_clicked(on_schedule)
r_tr.on_clicked(on_trend)

plt.show()
