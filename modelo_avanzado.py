# -*- coding: utf-8 -*-
"""
Bitcoin Halving Cycles — Modelo de Proyección via Spline GCV + ALS
==================================================================

Modelo formal:
    Y_n(d) = A(n) · φ(d / L(n)) + ε

donde:
    Y_n(d)  = log F(d,n) - log(100)     precio indexado centrado en log-espacio
    A(n)    = amplitud del ciclo         decreciente en n
    L(n)    = largo del ciclo en días    creciente en n, converge a L_inf
    φ : [0,1] → R                        forma universal, φ(0)=0, max(φ)=1
    ε       ~ ruido (autocorrelacionado — bootstrap por bloques)

Estimación (mínimos cuadrados alternados, ALS):
    1. Dado A(n): z = Y_n/A(n) se promedia por bins de t entre ciclos y se
       ajusta una spline cúbica suavizada con λ elegido por GCV
       (scipy.interpolate.make_smoothing_spline). El promediado por bins
       elimina los puntos casi-duplicados en t que rompen la spline directa.
    2. Dado φ: A(n) = Σφ(tᵢ)Y_n(dᵢ) / Σφ(tᵢ)² — regresión sin intercepto.
       Para el ciclo 2024 (censurado a la derecha) usa solo la parte
       observada: A(4) sale de los ~800 días de datos, no del máximo parcial.
    Iterar 1-2 hasta converger (φ se re-normaliza en cada paso, lo que fija
    la escala de A).

Incertidumbre:
    Bootstrap por bloques circulares de residuos (8 semanas por bloque):
    Y*_n = A(n)·φ(t) + e*_n, refit ALS completo en cada iteración.

Hipótesis asintótica (NO demostrada — extrapolación con 3 puntos):
    L(n) → L_inf ≈ 1458d (protocolo). Si además A(n) → 0, los ciclos
    desaparecen: F(d,n) → 100. Las familias pow y exp para el decaimiento
    de A(n)·L(n) ajustan igual de bien in-sample y divergen al extrapolar —
    se reportan ambas como escenarios.
"""

import os
import sys
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from scipy.interpolate import make_smoothing_spline, interp1d
from scipy.optimize import curve_fit
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. DATOS
# ─────────────────────────────────────────────────────────────────────────────

print("Descargando datos 2012-2014 (CoinMetrics)...")
r = requests.get(
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
    params={"assets": "btc", "metrics": "PriceUSD", "frequency": "1d",
            "start_time": "2012-01-01", "end_time": "2014-09-30",
            "page_size": 10000},
    timeout=30,
)
r.raise_for_status()
df_cm           = pd.DataFrame(r.json()["data"])
df_cm["date"]   = pd.to_datetime(df_cm["time"]).dt.tz_localize(None).dt.normalize()
df_cm["close"]  = pd.to_numeric(df_cm["PriceUSD"], errors="coerce")
df_cm           = df_cm[["date", "close"]].set_index("date").dropna()

# Verificación de datos: el pico de 2013 debe estar presente (~$1,150)
peak_2013 = df_cm.loc["2013-11-01":"2013-12-31", "close"].max()
print(f"  Precio máximo nov-dic 2013 en CoinMetrics: ${peak_2013:,.0f} (esperado ~$1,150)")

print("Descargando datos 2014-hoy (Yahoo Finance)...")
raw             = yf.download("BTC-USD", start="2014-09-01", interval="1d",
                               progress=False, auto_adjust=True)
df_yf           = raw[["Close"]].copy()
df_yf.columns  = ["close"]
df_yf.index    = pd.to_datetime(df_yf.index).tz_localize(None).normalize()

df = pd.concat([df_cm, df_yf]).sort_index()
df = df[~df.index.duplicated(keep="last")]
df = df.resample("W-MON").last().dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNDAMENTOS: L(n), series por ciclo
# ─────────────────────────────────────────────────────────────────────────────

HALVINGS_INFO = [
    {"nombre": "Ciclo 2012", "n": 1, "halving": pd.Timestamp("2012-11-28"), "color": "#ff7b72"},
    {"nombre": "Ciclo 2016", "n": 2, "halving": pd.Timestamp("2016-07-09"), "color": "#58a6ff"},
    {"nombre": "Ciclo 2020", "n": 3, "halving": pd.Timestamp("2020-05-11"), "color": "#3fb950"},
    {"nombre": "Ciclo 2024", "n": 4, "halving": pd.Timestamp("2024-04-19"), "color": "#f0b90b"},
]

# L(n) exacto de halvings conocidos
for i in range(3):
    HALVINGS_INFO[i]["L"] = (HALVINGS_INFO[i+1]["halving"] - HALVINGS_INFO[i]["halving"]).days

# L(4): modelo de convergencia L(n) = L_inf - c·exp(-λ·n)
# L_inf teórico: 210,000 bloques × 10 min/bloque = 1,458.33 días
L_INF   = 1458.33
L_known = np.array([h["L"] for h in HALVINGS_INFO[:3]], dtype=float)
n_known = np.array([1., 2., 3.])

def L_model(n, c, lam):
    return L_INF - c * np.exp(-lam * n)

try:
    popt_L, _ = curve_fit(L_model, n_known, L_known, p0=[300., 0.5], maxfev=5000)
    L4_est    = float(L_model(4., *popt_L))
    # Clamp a rango razonable: no puede superar L_inf ni ser menor que L(3)
    L4_est    = float(np.clip(L4_est, L_known[-1], L_INF))
except Exception:
    L4_est = float(L_INF - (L_INF - L_known[-1]) * 0.3)

HALVINGS_INFO[3]["L"] = round(L4_est)

print(f"\nL(n) [días]: {[h['L'] for h in HALVINGS_INFO]}")
print(f"L_inf teórico: {L_INF:.1f} días")

# Extraer series
LOG100 = np.log(100.0)

def nearest_price(date):
    idx = df.index.get_indexer([date], method="nearest")[0]
    return float(df.iloc[idx]["close"])

hoy = pd.Timestamp(datetime.today().date())

ciclos = []
for h in HALVINGS_INFO:
    halving     = h["halving"]
    L           = h["L"]
    p_base      = nearest_price(halving)
    mask        = (df.index >= halving) & (df.index <= halving + pd.Timedelta(days=L))
    sub         = df.loc[mask, "close"]
    dias        = (sub.index - halving).days.values.astype(float)
    F           = (sub.values / p_base) * 100.0
    Y           = np.log(F) - LOG100          # Y_n(d) = log F(d,n) - log(100)
    t           = dias / L                    # tiempo normalizado t ∈ [0,1]
    es_completo = (h["n"] < 4)

    ciclos.append({**h,
        "p_base": p_base, "dias": dias, "F": F,
        "Y": Y, "t": t,
        "es_completo": es_completo,
    })

dias_hoy = (hoy - HALVINGS_INFO[3]["halving"]).days
t_hoy    = dias_hoy / HALVINGS_INFO[3]["L"]
actual   = ciclos[3]

print(f"L(n): {[c['L'] for c in ciclos]}")
print(f"Hoy: d+{dias_hoy}  t_hoy={t_hoy:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTIMACIÓN DE φ Y A(n) — BINS + SPLINE GCV + MÍNIMOS CUADRADOS ALTERNADOS
# ─────────────────────────────────────────────────────────────────────────────
# Paso φ: cada ciclo aporta z = Y/A(n) sobre su grilla t. Se promedia z por
#   bins de t (primero dentro de cada ciclo, después entre ciclos con peso =
#   nº de ciclos con datos en el bin) y se ajusta una spline cúbica suavizada
#   con λ por GCV. Los bins eliminan los t casi-duplicados entre ciclos que
#   hacían explotar la spline directa; GCV elimina la elección manual de s.
# Paso A: regresión sin intercepto de Y_n sobre φ(t). Para el ciclo 2024
#   usa solo la parte observada — resuelve la censura sin usar el máximo.

N_BINS = 80

def binned_phi(ciclos_train, A_de):
    """Estima φ dado A(n). Retorna callable φ: [0,1] → R, φ(0)=0, max(φ)=1."""
    edges   = np.linspace(0., 1., N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medias  = np.full((len(ciclos_train), N_BINS), np.nan)

    for j, c in enumerate(ciclos_train):
        z   = c["Y"] / A_de[c["n"]]
        idx = np.clip(np.digitize(c["t"], edges) - 1, 0, N_BINS - 1)
        for b in np.unique(idx):
            medias[j, b] = z[idx == b].mean()

    z_bin = np.nanmean(medias, axis=0)
    w_bin = np.sum(~np.isnan(medias), axis=0).astype(float)
    ok    = w_bin > 0

    spline = make_smoothing_spline(centers[ok], z_bin[ok], w=w_bin[ok])

    # Constraints φ(0)=0, max(φ)=1
    grid        = np.linspace(0., 1., 500)
    phi_0       = float(spline(0.))
    phi_max_val = float(np.max(spline(grid)))
    escala      = phi_max_val - phi_0
    if abs(escala) < 1e-8:
        escala = 1.0

    def phi(t_in):
        t_clipped = np.clip(np.asarray(t_in, dtype=float), 0., 1.)
        return (spline(t_clipped) - phi_0) / escala

    return phi

def ajustar_A(c, phi):
    """A(n) = argmin_A Σ(Y - A·φ(t))² — regresión sin intercepto."""
    f = phi(c["t"])
    return float(f @ c["Y"] / (f @ f))

def ajustar_modelo(ciclos_train, n_iter=6):
    """ALS: alterna estimación de φ (dado A) y de A (dado φ)."""
    A_de = {c["n"]: float(np.max(c["Y"])) for c in ciclos_train}  # init: max
    for _ in range(n_iter):
        phi = binned_phi(ciclos_train, A_de)
        A_de = {c["n"]: ajustar_A(c, phi) for c in ciclos_train}
    return phi, A_de

# Fit final: ciclos 1-3 completos + ciclo 4 parcial hasta t_hoy
ciclos_completos = [c for c in ciclos if c["es_completo"]]
mask_obs         = actual["t"] <= t_hoy
ciclo4_parcial   = {**actual,
                    "t": actual["t"][mask_obs],
                    "Y": actual["Y"][mask_obs]}
ciclos_fit       = ciclos_completos + [ciclo4_parcial]

print("\nAjustando φ y A(n) por mínimos cuadrados alternados...")
phi_hat, A_hat = ajustar_modelo(ciclos_fit)

for c in ciclos:
    c["A"] = A_hat[c["n"]]

A_vals = np.array([c["A"] for c in ciclos])
L_vals = np.array([c["L"] for c in ciclos], dtype=float)
print(f"A(n) estimados (ALS): {[round(a, 4) for a in A_vals]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDACIÓN: entrenar en ciclos 1+2, predecir ciclo 3 — CON BASELINES
# ─────────────────────────────────────────────────────────────────────────────
# La amplitud A(3) se ajusta por regresión para las tres formas candidatas —
# la comparación mide solo la calidad de la FORMA φ, que es lo que el modelo
# aporta. Baselines: recta φ(t)=t y la forma del ciclo 2016 reescalada.

print("\nValidación out-of-sample: train={1,2} → predice ciclo 3 completo...")
phi_val, A_val = ajustar_modelo(ciclos_completos[:2])
c3 = ciclos_completos[2]

def rmse_forma(forma_en_t3):
    """RMSE de una forma candidata con amplitud ajustada por regresión."""
    f  = np.asarray(forma_en_t3, dtype=float)
    A3 = float(f @ c3["Y"] / (f @ f))
    z_true = c3["Y"] / A3
    return float(np.sqrt(np.mean((z_true - f) ** 2))), A3, z_true

# Modelo
z3_pred                  = phi_val(c3["t"])
rmse_val, A3_val, z3_true = rmse_forma(z3_pred)
corr_val                 = float(np.corrcoef(z3_true, z3_pred)[0, 1])
mae_val                  = float(np.mean(np.abs(z3_true - z3_pred)))
r2_val                   = float(1 - np.sum((z3_true - z3_pred) ** 2)
                                   / np.sum((z3_true - np.mean(z3_true)) ** 2))
# MAPE se omite deliberadamente: z ≈ 0 al inicio del ciclo lo hace explotar

# Baseline 1: recta φ(t) = t
rmse_recta, _, _ = rmse_forma(c3["t"])

# Baseline 2: forma del ciclo anterior (2016) interpolada, reescalada
c2      = ciclos_completos[1]
z2      = c2["Y"] / A_val[2]
shape2  = interp1d(c2["t"], z2, bounds_error=False,
                   fill_value=(z2[0], z2[-1]))(c3["t"])
rmse_c2, _, _ = rmse_forma(shape2)

print(f"  RMSE forma — modelo φ̂        : {rmse_val:.4f}  (r={corr_val:.4f})")
print(f"  RMSE forma — recta φ(t)=t    : {rmse_recta:.4f}")
print(f"  RMSE forma — ciclo 2016 crudo: {rmse_c2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DECAIMIENTO A(n) — dos escenarios, train solo en n=1,2,3
# ─────────────────────────────────────────────────────────────────────────────
# Con 3 puntos y 2 parámetros no se puede distinguir pow de exp: se reportan
# ambos como escenarios, sin promediarlos.

n_vals      = np.array([c["n"] for c in ciclos], dtype=float)
n_fit_decay = n_vals[:3]
A_fit_decay = A_vals[:3]

def decay_A_pow(n, a, b):
    return a * n ** (-b)

def decay_A_exp(n, a, b):
    return a * np.exp(-b * (n - 1))

popt_pow, _ = curve_fit(decay_A_pow, n_fit_decay, A_fit_decay, p0=[6., 0.8], maxfev=5000)
popt_exp, _ = curve_fit(decay_A_exp, n_fit_decay, A_fit_decay, p0=[6., 0.4], maxfev=5000)
A4_pred_pow = float(decay_A_pow(4., *popt_pow))
A4_pred_exp = float(decay_A_exp(4., *popt_exp))

# Para la proyección se usa A(4) estimado por ALS sobre los ~800 días
# observados del ciclo — no la extrapolación del decaimiento.
A4_used = A_hat[4]

print(f"\nDecaimiento A(n) (train n=1,2,3) — escenarios:")
print(f"  Power law   → A(4): {A4_pred_pow:.4f}")
print(f"  Exponencial → A(4): {A4_pred_exp:.4f}")
print(f"  A(4) estimado de los datos observados (ALS): {A4_used:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PROYECCIÓN CICLO 4
# ─────────────────────────────────────────────────────────────────────────────

precio_actual = nearest_price(hoy)

t_future = np.linspace(t_hoy, 1.0, 300)
d_future = t_future * actual["L"]

# Proyección anclada al precio observado hoy.
# El modelo predice el CAMBIO desde t_hoy, no el nivel absoluto.
# Y_proj(d) = Y_obs(hoy) + A(4) · [φ(d/L(4)) - φ(t_hoy)]
Y_hoy_obs = float(np.log(precio_actual / actual["p_base"]))
delta_phi = phi_hat(t_future) - float(phi_hat(t_hoy))
Y_proj    = Y_hoy_obs + A4_used * delta_phi
F_proj    = actual["p_base"] * np.exp(Y_proj)

# Fondo: mínimo de Y_proj en t > t_hoy
i_fondo      = int(np.argmin(Y_proj))
d_fondo      = float(d_future[i_fondo])
fecha_fondo  = actual["halving"] + pd.Timedelta(days=int(d_fondo))
precio_fondo = float(F_proj[i_fondo])

precio_pico      = actual["p_base"] * float(np.exp(np.max(actual["Y"])))  # pico observado
caida_desde_pico = (precio_actual / precio_pico - 1) * 100
caida_restante   = (precio_fondo  / precio_actual - 1) * 100

# ── Modelo simple (base.py): promedio de días y caída histórica ───────────────
# Replica exacta de la lógica de base.py para comparación directa
simple_stats = []
for c in ciclos_completos:
    # Mismo criterio que base.py: buscar pico en el primer 60% del ciclo
    corte       = int(len(c["F"]) * 0.6)
    pos_max     = int(np.argmax(c["F"][:corte]))
    dias_pico   = int(c["dias"][pos_max])
    nivel_pico  = float(c["F"][pos_max])
    post_pico   = c["F"][pos_max:]
    pos_fondo   = pos_max + int(np.argmin(post_pico))
    dias_fondo  = int(c["dias"][pos_fondo])
    nivel_fondo = float(c["F"][pos_fondo])
    simple_stats.append({"dias_pico": dias_pico, "nivel_pico": nivel_pico,
                          "dias_fondo": dias_fondo, "nivel_fondo": nivel_fondo})

dias_pico_avg  = round(np.mean([s["dias_pico"]  for s in simple_stats]))
dias_fondo_avg = round(np.mean([s["dias_fondo"] for s in simple_stats]))
caida_avg      = np.mean([s["nivel_fondo"] / s["nivel_pico"] for s in simple_stats])

# Pico del ciclo actual (observado)
pos_max_actual   = int(np.argmax(actual["F"]))
dias_pico_actual = int(actual["dias"][pos_max_actual])
idx_pico_actual  = float(actual["F"][pos_max_actual])

d_fondo_simple      = dias_pico_actual + (dias_fondo_avg - dias_pico_avg)
idx_fondo_simple    = idx_pico_actual * caida_avg
precio_fondo_simple = actual["p_base"] * idx_fondo_simple / 100
fecha_fondo_simple  = actual["halving"] + pd.Timedelta(days=int(d_fondo_simple))

# ─────────────────────────────────────────────────────────────────────────────
# 6. BOOTSTRAP POR BLOQUES DE RESIDUOS: banda de incertidumbre 80%
# ─────────────────────────────────────────────────────────────────────────────
# Con solo 3 ciclos completos, remuestrear ciclos enteros no tiene soporte
# (10 multiconjuntos posibles). En su lugar: bootstrap circular por bloques
# de los residuos e = Y - A·φ(t) de cada ciclo (bloques de 8 semanas para
# respetar la autocorrelación), se generan ciclos sintéticos
# Y* = A·φ(t) + e* y se re-ajusta el modelo ALS completo en cada iteración.
# Sin try/except: si el fit falla, debe verse.

print("\nBootstrap por bloques (N=1000)...")
N_BOOT = 1000
BLOCK  = 8   # semanas ≈ 2 meses
rng    = np.random.default_rng(42)

residuos = {c["n"]: c["Y"] - A_hat[c["n"]] * phi_hat(c["t"]) for c in ciclos_fit}

def remuestrear_bloques(e, rng):
    """Bootstrap circular por bloques de una serie de residuos."""
    m   = len(e)
    out = np.empty(m)
    pos = 0
    while pos < m:
        start = int(rng.integers(0, m))
        take  = min(BLOCK, m - pos)
        out[pos:pos + take] = e[(start + np.arange(take)) % m]
        pos += take
    return out

F_boot   = []
fondos_b = []
for _ in range(N_BOOT):
    sintticos = []
    for c in ciclos_fit:
        e_star = remuestrear_bloques(residuos[c["n"]], rng)
        Y_star = A_hat[c["n"]] * phi_hat(c["t"]) + e_star
        sintticos.append({"n": c["n"], "t": c["t"], "Y": Y_star})

    phi_b, A_b  = ajustar_modelo(sintticos, n_iter=3)
    delta_phi_b = phi_b(t_future) - float(phi_b(t_hoy))
    Y_b         = Y_hoy_obs + A_b[4] * delta_phi_b
    F_b         = actual["p_base"] * np.exp(Y_b)
    F_boot.append(F_b)
    fondos_b.append(float(F_b[int(np.argmin(Y_b))]))

if not fondos_b:
    raise RuntimeError("Bootstrap sin iteraciones válidas — revisar modelo")

F_boot   = np.array(F_boot)
F_lo     = np.percentile(F_boot, 10, axis=0)
F_hi     = np.percentile(F_boot, 90, axis=0)
fondo_lo = float(np.percentile(fondos_b, 10))
fondo_hi = float(np.percentile(fondos_b, 90))

# ─────────────────────────────────────────────────────────────────────────────
# 7. HIPÓTESIS ASINTÓTICA: área A(n)·L(n) — dos escenarios
# ─────────────────────────────────────────────────────────────────────────────

area_vals = A_vals * L_vals
area_fit  = A_fit_decay * L_vals[:3]

def decay_exp(n, alpha, beta):
    return alpha * np.exp(-beta * n)

def decay_pow(n, alpha, beta):
    return alpha * n ** (-beta)

popt_ae, _ = curve_fit(decay_exp, n_fit_decay, area_fit, p0=[8000., 0.3], maxfev=5000)
popt_ap, _ = curve_fit(decay_pow, n_fit_decay, area_fit, p0=[6000., 0.8], maxfev=5000)
n_ext      = np.linspace(1, 14, 300)
area_exp   = decay_exp(n_ext, *popt_ae)
area_pow   = decay_pow(n_ext, *popt_ap)
# Ciclo en que el escenario exponencial cae al 1% del área inicial
n_zero     = float(-np.log(0.01 * area_fit[0] / popt_ae[0]) / popt_ae[1])

# ─────────────────────────────────────────────────────────────────────────────
# 8. CONSOLA: resumen completo
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  MODELO: Y_n(d) = A(n)·φ(d/L(n))")
print("  Estimación: bins + spline GCV + mínimos cuadrados alternados")
print("=" * 64)
print(f"\n  Hoy               : d+{dias_hoy} ({hoy.date()})")
print(f"  Precio actual     : ${precio_actual:>10,.0f}")
print(f"  Precio en el pico : ${precio_pico:>10,.0f}")
print(f"  Caída desde pico  : {caida_desde_pico:>+7.1f}%")
print(f"\n  ── COMPARACIÓN DE MODELOS ──")
print(f"  {'Modelo':<30} {'Día':>6}  {'Fecha':<12}  {'Precio':>10}  {'Caída':>7}")
print(f"  {'-'*68}")
print(f"  {'Simple (promedio histórico)':<30} {'d+'+str(int(d_fondo_simple)):>6}  {fecha_fondo_simple.strftime('%Y-%m-%d'):<12}  ${precio_fondo_simple:>9,.0f}  {(precio_fondo_simple/precio_actual-1)*100:>+6.1f}%")
print(f"  {'Avanzado (A4='+format(A4_used, '.2f')+')':<30} {'d+'+str(int(d_fondo)):>6}  {fecha_fondo.strftime('%Y-%m-%d'):<12}  ${precio_fondo:>9,.0f}  {caida_restante:>+6.1f}%")
print(f"  Banda 80% del fondo (bootstrap): ${fondo_lo:,.0f} – ${fondo_hi:,.0f}")
print(f"\n  ── VALIDACIÓN (out-of-sample ciclo 3, amplitud ajustada) ──")
print(f"  Modelo φ̂          : RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  R²={r2_val:.4f}  r={corr_val:.4f}")
print(f"  RMSE recta φ(t)=t : {rmse_recta:.4f}")
print(f"  RMSE ciclo 2016   : {rmse_c2:.4f}")
print(f"\n  ── HIPÓTESIS ASINTÓTICA (extrapolación con 3 puntos) ──")
print(f"  A(n) estimados  : {[round(a, 3) for a in A_vals]}")
print(f"  A(4) escenarios : pow={A4_pred_pow:.4f}, exp={A4_pred_exp:.4f}  |  datos={A4_used:.4f}")
print(f"  L(n): {[int(l) for l in L_vals]}")
print(f"  Área A(n)·L(n): {[round(a, 0) for a in area_vals]}")
print(f"  Escenario exp: área cae al 1% en ciclo ~{n_zero:.1f}")
print(f"  Escenario pow: decae lento, nunca llega a 0")

# ─────────────────────────────────────────────────────────────────────────────
# 9. GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

DARK  = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
TEXT  = "#c9d1d9"
WHITE = "#f0f6fc"

def style_ax(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(DARK)
gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

grid_t   = np.linspace(0., 1., 400)
phi_vals = phi_hat(grid_t)

# ── Panel A: verificación colapso φ ──────────────────────────────────────────
ax_phi = fig.add_subplot(gs[0, 0])
style_ax(ax_phi)

for c in ciclos:
    t_plot = c["t"] if c["es_completo"] else c["t"][c["t"] <= t_hoy]
    z_plot = (c["Y"] / c["A"])[:len(t_plot)]
    ax_phi.plot(t_plot, z_plot, color=c["color"], alpha=0.75, lw=1.5, label=c["nombre"])

ax_phi.plot(grid_t, phi_vals, color=WHITE, lw=2.2, ls="--", label="φ̂ estimada", zorder=5)
ax_phi.axvline(t_hoy, color="#f0b90b", lw=1, ls=":", alpha=0.7,
               label=f"Hoy (t={t_hoy:.2f})")
ax_phi.axhline(0, color=GRID, lw=0.8)
ax_phi.axhline(1, color=GRID, lw=0.8, ls=":")
ax_phi.set_title("Panel A — Verificación: colapso en φ\n"
                 "Curvas normalizadas Y_n/A(n) vs t=d/L(n)",
                 color=WHITE, fontsize=10)
ax_phi.set_xlabel("t = d / L(n)  (tiempo normalizado)", color=TEXT)
ax_phi.set_ylabel("Y_n(d) / A(n)  ≈  φ(t)", color=TEXT)
ax_phi.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# ── Panel B: validación out-of-sample con baselines ──────────────────────────
ax_val = fig.add_subplot(gs[0, 1])
style_ax(ax_val)

ax_val.plot(c3["t"], z3_true, color="#3fb950", lw=2, label="Ciclo 2020 real")
ax_val.plot(c3["t"], z3_pred, color=WHITE, lw=1.8, ls="--",
            label=f"Modelo φ̂ (train 2012+2016)\nRMSE={rmse_val:.4f}  r={corr_val:.4f}")
ax_val.plot(c3["t"], c3["t"], color="#8b949e", lw=1.2, ls=":",
            label=f"Baseline recta  RMSE={rmse_recta:.4f}")
ax_val.plot(c3["t"], shape2, color="#58a6ff", lw=1.2, ls=":", alpha=0.8,
            label=f"Baseline ciclo 2016  RMSE={rmse_c2:.4f}")
ax_val.axhline(0, color=GRID, lw=0.8)
ax_val.set_title("Panel B — Validación out-of-sample\n"
                 "Train: ciclos 1+2  →  Predice ciclo 3 completo (amplitud ajustada)",
                 color=WHITE, fontsize=10)
ax_val.set_xlabel("t = d / L(n)", color=TEXT)
ax_val.set_ylabel("Y / A  ≈  φ(t)", color=TEXT)
ax_val.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# ── Panel C: proyección ciclo 4 — escala indexada (igual que base.py) ────────
ax_proj = fig.add_subplot(gs[1, 0])
style_ax(ax_proj)

F_idx_proj  = F_proj / actual["p_base"] * 100.0
F_idx_lo    = F_lo   / actual["p_base"] * 100.0
F_idx_hi    = F_hi   / actual["p_base"] * 100.0
F_idx_fondo = precio_fondo / actual["p_base"] * 100.0

# También graficar los 3 ciclos históricos para contexto
for c in ciclos:
    alpha = 0.5 if c["n"] < 4 else 1.0
    lw    = 1.3 if c["n"] < 4 else 2.2
    ax_proj.plot(c["dias"], c["F"], color=c["color"], lw=lw, alpha=alpha,
                 label=c["nombre"])

# Proyección indexada
ax_proj.plot(d_future, F_idx_proj,
             color="#f0b90b", lw=2, ls="--", label="Proyección φ̂")
ax_proj.fill_between(d_future, F_idx_lo, F_idx_hi,
                     color="#f0b90b", alpha=0.15, label="Banda 80%")

# Fondo
ax_proj.scatter(d_fondo, F_idx_fondo,
                color="#f0b90b", s=140, marker="v", zorder=6,
                edgecolors=DARK, lw=1.5)
ax_proj.annotate(
    f"Fondo\nd+{int(d_fondo)}  ·  {fecha_fondo.strftime('%Y-%m-%d')}\n"
    f"${precio_fondo:,.0f}  [{fondo_lo:,.0f}–{fondo_hi:,.0f}]",
    xy=(d_fondo, F_idx_fondo),
    xytext=(30, 25), textcoords="offset points",
    color="#f0b90b", fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color="#f0b90b", lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor="#f0b90b", alpha=0.9)
)

# Modelo simple: punto único con línea vertical
ax_proj.axvline(d_fondo_simple, color="#a371f7", lw=1.2, ls=":", alpha=0.8)
ax_proj.scatter(d_fondo_simple,
                precio_fondo_simple / actual["p_base"] * 100,
                color="#a371f7", s=120, marker="v", zorder=6,
                edgecolors=DARK, lw=1.5,
                label=f"Simple: d+{int(d_fondo_simple)} ${precio_fondo_simple:,.0f}")

ax_proj.axvline(dias_hoy, color=WHITE, lw=1.2, ls="-.", alpha=0.6,
                label=f"Hoy d+{dias_hoy}")
ax_proj.set_yscale("log")
ax_proj.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}x"))
ax_proj.set_title("Panel C — Proyección ciclo 2024\n"
                  f"Fondo: d+{int(d_fondo)}  ·  {fecha_fondo.strftime('%Y-%m-%d')}  ·  "
                  f"${precio_fondo:,.0f}",
                  color=WHITE, fontsize=10)
ax_proj.set_xlabel("Días desde el halving", color=TEXT)
ax_proj.set_ylabel("Precio indexado (halving = 100x)", color=TEXT)
ax_proj.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7.5)

# ── Panel D: hipótesis asintótica — área A(n)·L(n), dos escenarios ───────────
ax_asint = fig.add_subplot(gs[1, 1])
style_ax(ax_asint)

nombres = ["2012\n(n=1)", "2016\n(n=2)", "2020\n(n=3)", "2024\n(n=4)"]
bars = ax_asint.bar(n_vals, area_vals,
                    color=[c["color"] for c in ciclos],
                    alpha=0.85, width=0.5, zorder=3)

for bar, val in zip(bars, area_vals):
    ax_asint.text(bar.get_x() + bar.get_width() / 2, val + 50,
                  f"{val:.0f}", ha="center", color=TEXT, fontsize=8)

ax_asint.plot(n_ext, area_exp, color=WHITE, lw=1.8, ls="--", alpha=0.8,
              label=f"Escenario exp: ~1% en ciclo ~{n_zero:.0f}")
ax_asint.plot(n_ext, area_pow, color="#8b949e", lw=1.8, ls="--", alpha=0.8,
              label="Escenario pow: decae lento, no llega a 0")
ax_asint.axhline(0, color=GRID, lw=0.8)
ax_asint.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID,
                labelcolor=TEXT, fontsize=8)

ax_asint.set_xticks(n_vals)
ax_asint.set_xticklabels(nombres, color=TEXT)
ax_asint.set_xlim(0.5, 5.0)
ax_asint.set_ylim(0, max(area_vals) * 1.45)   # espacio libre arriba para leyenda y nota
ax_asint.set_title("Panel D — Hipótesis asintótica\n"
                   "Decaimiento del área A(n)·L(n): dos escenarios compatibles",
                   color=WHITE, fontsize=10)
ax_asint.set_xlabel("Ciclo n", color=TEXT)
ax_asint.set_ylabel("Área = A(n) · L(n)", color=TEXT)

nota_text = (
    "Nota:\n"
    f"L(n) → L∞={L_INF:.0f}d  (protocolo)\n"
    "⟹  φ(d/L(n)) → φ(d/L∞)\n"
    "Con 3 puntos, pow y exp ajustan\n"
    "igual in-sample y divergen al\n"
    "extrapolar: es un rango, no\n"
    "una certeza."
)
ax_asint.text(0.97, 0.97, nota_text,
              transform=ax_asint.transAxes,
              va="top", ha="right",
              color=TEXT, fontsize=7.5, fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                        edgecolor=GRID, alpha=0.9))

# ── Título global ─────────────────────────────────────────────────────────────
fig.suptitle(
    f"Bitcoin — Modelo Y_n(d)=A(n)·φ(d/L(n))  |  "
    f"Fondo ciclo 2024: d+{int(d_fondo)} · {fecha_fondo.strftime('%Y-%m-%d')} · "
    f"${precio_fondo:,.0f}  [{fondo_lo:,.0f}–{fondo_hi:,.0f}]",
    color=WHITE, fontsize=11, y=0.995
)

os.makedirs("figures", exist_ok=True)
fecha_export   = datetime.today().strftime("%Y%m%d_%H%M")
nombre_archivo = os.path.join("figures", f"bitcoin_modelo_avanzado_{fecha_export}.png")
plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.savefig(os.path.join("figures", "modelo_avanzado_latest.png"),
            dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nGráfico guardado: {nombre_archivo} (+ figures/modelo_avanzado_latest.png)")
plt.show()
