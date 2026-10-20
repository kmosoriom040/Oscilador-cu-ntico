"""
Figuras del póster "péndulo cuántico".
fig_base.pdf : espectro del péndulo cuántico (U0 = 2), recalculado con funciones de Mathieu
               (Baker, Blackburn & Smith, Am. J. Phys. 70, 525 (2002), Eqs. 4-6 y Tabla I).
fig_caos.pdf : propuesta — crecimiento de una diferencia inicial de 1e-14 rad
               (la incertidumbre cuántica estimada en el artículo) en el péndulo forzado
               (caótico) y en el péndulo libre de gran amplitud.
"""
import numpy as np
from scipy.special import mathieu_a, mathieu_b
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, C3 = "#6A3DB8", "#D0509A", "#4F8FD8"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#e3e3e3"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 17, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "mathtext.fontset": "dejavusans"})
def estilo(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------------- Figura 1: espectro ----------------
U0, q = 2.0, 4.0
lab = [("a",0),("b",2),("a",2),("b",4),("a",4),("b",6),("a",6),("b",8),("a",8),("b",10),("a",10)]
eps = np.array([(mathieu_a(i,q) if t=="a" else mathieu_b(i,q))/4 + U0 for t,i in lab])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"wspace": 0.22})
th = np.linspace(-np.pi, np.pi, 400)
a1.plot(th, U0*(1-np.cos(th)), color=C1, lw=2.6)
a1.plot(th, U0*th**2/2, color=C3, lw=2.2, ls="--")
for e in eps[eps < 7.2]:
    x = np.arccos(1 - e/U0) if e < 2*U0 else np.pi
    a1.plot([-x, x], [e, e], color=INK, lw=1.6)
a1.axhline(2*U0, color=MUTED, ls=":", lw=1.4)
a1.text(-3.05, 2*U0+0.15, "barrera $2U_0$", color=MUTED, fontsize=14)
a1.text(1.25, 1.2, r"$U_0(1-\cos\theta)$", color=C1, fontsize=15)
a1.text(1.35, 5.6, r"$U_0\theta^2/2$", color=C3, fontsize=15)
a1.set_xlim(-np.pi, np.pi); a1.set_ylim(0, 7.5)
a1.set_xticks([-np.pi, 0, np.pi]); a1.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
a1.set_xlabel(r"$\theta$"); a1.set_ylabel(r"energía $\varepsilon$")
a1.set_title("(a) Pozo y niveles ($U_0=2$)", color=INK, fontsize=17, fontweight="bold")
n = np.arange(len(eps))
a2.plot(n, (2*n+1)*np.sqrt(U0/2), color=C3, ls="--", lw=2.2)
m = np.array([0,2,2,4,4,6,6,8,8,10,10])
a2.plot(n[5:], m[5:]**2/4 + U0, color=C2, ls=":", lw=2.6)
a2.plot(n, eps, "o", color=C1, ms=9)
a2.text(6.9, 8.2, "oscilador\narmónico", color=C3, fontsize=14)
a2.text(6.6, 22.5, "rotor libre", color=C2, fontsize=14)
a2.text(3.9, 2.0, "péndulo", color=C1, fontsize=14)
a2.set_xlabel("índice del estado"); a2.set_ylabel(r"$\varepsilon$")
a2.set_title("(b) Del oscilador al rotor", color=INK, fontsize=17, fontweight="bold")
for ax in (a1, a2): estilo(ax)
fig.savefig("fig_base.pdf", bbox_inches="tight"); fig.savefig("fig_base.png", dpi=150, bbox_inches="tight")

# ---------------- Figura 2: caos ----------------
QD, G, WD = 2.0, 1.5, 2/3
def rhs(t, Y, qd, g, wd):
    th, w, d, dw = Y
    return [w, -w/qd - np.sin(th) + g*np.cos(wd*t), dw, -dw/qd - 2*np.cos(th+d/2)*np.sin(d/2)]
s = solve_ivp(lambda t,y:[y[1], -y[1]/QD-np.sin(y[0])+G*np.cos(WD*t)], (0,300), [0.2,0],
              rtol=1e-11, atol=1e-12, method="DOP853")
T = 450
te = np.linspace(300, 300+T, 5000)
c = solve_ivp(rhs, (300, 300+T), [*s.y[:,-1], 1e-14, 0], args=(QD,G,WD), t_eval=te,
              method="DOP853", rtol=1e-11, atol=1e-30)
tf = np.linspace(0, T, 5000)
l = solve_ivp(rhs, (0, T), [2.5, 0, 1e-14, 0], args=(1e12, 0.0, 1.0), t_eval=tf,
              method="DOP853", rtol=1e-11, atol=1e-30)
tc, dc = c.t-300, np.abs(c.y[2]); tstar = tc[np.argmax(dc > 1)]
fig, ax = plt.subplots(figsize=(13.4, 4.6))
ax.semilogy(tc, dc, color=C2, lw=2.0)
ax.semilogy(l.t, np.maximum.accumulate(np.abs(l.y[2])), color=C3, lw=2.6)
ax.axhline(1e-14, color=MUTED, ls=":", lw=1.4); ax.axhline(1, color=MUTED, ls=":", lw=1.4)
ax.axvline(tstar, color=INK, ls="--", lw=1.4)
ax.text(5, 1.7e-15, r"incertidumbre cuántica $\Delta\theta\approx10^{-14}$ rad [1]", color=INK, fontsize=14)
ax.text(5, 2.0, "escala macroscópica (1 rad)", color=INK, fontsize=14)
ax.text(tstar+6, 1e-6, rf"$t^*\approx{tstar:.0f}/\omega_0$" + "\n" + "(en este ejemplo)", color=INK, fontsize=14)
ax.text(120, 3e-6, "forzado (caótico)", color=C2, fontsize=15, rotation=0)
ax.text(255, 2e-11, "libre, gran amplitud (no caótico)", color=C3, fontsize=15)
ax.set_ylim(8e-16, 30); ax.set_xlim(0, T)
ax.set_xlabel(r"tiempo $\omega_0 t$"); ax.set_ylabel(r"$|\delta\theta(t)|$ (rad)")
ax.text(0.995, 0.02, "curva libre: envolvente", transform=ax.transAxes, ha="right", color=MUTED, fontsize=11)
estilo(ax)
fig.savefig("fig_caos.pdf", bbox_inches="tight"); fig.savefig("fig_caos.png", dpi=150, bbox_inches="tight")
print("t* =", tstar)
