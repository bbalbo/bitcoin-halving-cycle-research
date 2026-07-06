# -*- coding: utf-8 -*-
import sys, requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ── Datos ─────────────────────────────────────────────────────────────────────
print("Descargando datos 2012-2014 (CoinMetrics)...")
r = requests.get(
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
    params={"assets": "btc", "metrics": "PriceUSD", "frequency": "1d",
            "start_time": "2012-01-01", "end_time": "2014-09-30"},
    timeout=30,
)
r.raise_for_status()
df_cm = pd.DataFrame(r.json()["data"])
df_cm["date"]  = pd.to_datetime(df_cm["time"]).dt.tz_localize(None).dt.normalize()
df_cm["close"] = df_cm["PriceUSD"].astype(float)
df_cm = df_cm[["date", "close"]].set_index("date")

print("Descargando datos 2014-hoy (Yahoo Finance)...")
raw = yf.download("BTC-USD", start="2014-09-01", interval="1d",
                  progress=False, auto_adjust=True)
df_yf = raw[["Close"]].copy()
df_yf.columns = ["close"]
df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None).normalize()

df = pd.concat([df_cm, df_yf]).sort_index()
df = df[~df.index.duplicated(keep="last")]
df = df.resample("W-MON").last().dropna()

# ── Definicion de ciclos (solo halvings hardcodeados) ─────────────────────────
HALVINGS = [
    {"nombre": "Ciclo 2012", "halving": "2012-11-28", "color": "#ff7b72"},
    {"nombre": "Ciclo 2016", "halving": "2016-07-09", "color": "#58a6ff"},
    {"nombre": "Ciclo 2020", "halving": "2020-05-11", "color": "#3fb950"},
    {"nombre": "Ciclo 2024-? (actual)", "halving": "2024-04-19", "color": "#f0b90b"},
]
for c in HALVINGS:
    c["halving"] = pd.Timestamp(c["halving"])

# ventana de cada ciclo = dias hasta el siguiente halving (o promedio para el actual)
for i, c in enumerate(HALVINGS):
    if i + 1 < len(HALVINGS):
        c["ventana_dias"] = (HALVINGS[i + 1]["halving"] - c["halving"]).days
duraciones = [c["ventana_dias"] for c in HALVINGS if "ventana_dias" in c]
HALVINGS[-1]["ventana_dias"] = round(np.mean(duraciones))

CICLOS = HALVINGS

# ── Construir series indexadas (halving = 100) ────────────────────────────────
def nearest_price(date):
    idx = df.index.get_indexer([date], method="nearest")[0]
    return df.iloc[idx]["close"]

series = []
for i, c in enumerate(CICLOS):
    h = c["halving"]
    ventana = c["ventana_dias"]
    precio_base = nearest_price(h)
    mask = (df.index >= h) & (df.index <= h + pd.Timedelta(days=ventana))
    sub  = df.loc[mask, "close"]
    dias = (sub.index - h).days.values
    idx  = (sub.values / precio_base) * 100

    es_actual = (i == len(CICLOS) - 1)

    # pico = maximo en el primer 60% de la ventana (evita capturar el rally del siguiente ciclo)
    corte = int(len(idx) * 0.6)
    zona_pico = idx[:corte] if not es_actual else idx
    pos_max   = int(np.argmax(zona_pico))
    dias_pico = int(dias[pos_max])
    idx_pico  = float(idx[pos_max])

    if not es_actual:
        # fondo = minimo despues del pico
        post_pico  = idx[pos_max:]
        pos_fondo  = pos_max + int(np.argmin(post_pico))
        dias_fondo = int(dias[pos_fondo])
        idx_fondo  = float(idx[pos_fondo])
    else:
        dias_fondo = None
        idx_fondo  = None

    series.append({**c,
        "precio_base": precio_base,
        "dias": dias, "idx": idx,
        "dias_pico": dias_pico, "idx_pico": idx_pico,
        "dias_fondo": dias_fondo, "idx_fondo": idx_fondo,
    })

# ── Proyeccion ciclo actual ───────────────────────────────────────────────────
hist = [s for s in series if s["dias_fondo"] is not None]
dias_pico_avg  = round(np.mean([s["dias_pico"]  for s in hist]))
dias_fondo_avg = round(np.mean([s["dias_fondo"] for s in hist]))
caida_avg      = np.mean([s["idx_fondo"] / s["idx_pico"] for s in hist])

actual = series[-1]   # siempre el ciclo 2024
hoy = pd.Timestamp(datetime.today().date())
dias_hoy = (hoy - actual["halving"]).days

dias_fondo_proy = actual["dias_pico"] + (dias_fondo_avg - dias_pico_avg)
fecha_fondo_proy = actual["halving"] + pd.Timedelta(days=int(dias_fondo_proy))
idx_fondo_proy  = actual["idx_pico"] * caida_avg
precio_fondo_proy = actual["precio_base"] * idx_fondo_proy / 100
precio_actual   = nearest_price(hoy)
caida_actual    = (precio_actual / (actual["precio_base"] * actual["idx_pico"] / 100) - 1) * 100
caida_total_proy = (caida_avg - 1) * 100
dias_al_fondo   = (fecha_fondo_proy - hoy).days
ya_en_fondo     = dias_al_fondo <= 0

# ── Consola ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  COMPARACION PARALELA DE CICLOS")
print("=" * 62)
for s in series:
    p = s["precio_base"] * s["idx_pico"] / 100
    print(f"\n  {s['nombre']}")
    print(f"    Base halving : ${s['precio_base']:>10,.0f}  |  Pico dia +{s['dias_pico']}  ${p:>10,.0f}  ({s['idx_pico']:.0f}x)")
    if s["dias_fondo"]:
        f = s["precio_base"] * s["idx_fondo"] / 100
        print(f"    Fondo        : dia +{s['dias_fondo']}  ${f:>10,.0f}  ({(caida_avg-1)*100:.0f}% desde pico)")

print(f"\n  -- CICLO ACTUAL (Halving 2024-04-19) --")
print(f"  Pico detectado   : dia +{actual['dias_pico']}  ${actual['precio_base']*actual['idx_pico']/100:,.0f}")
print(f"  Hoy              : dia +{dias_hoy}  ${precio_actual:,.0f}  ({caida_actual:.1f}% desde pico)")
print(f"  Fondo proyectado : dia +{int(dias_fondo_proy)}  ({fecha_fondo_proy.date()})  ~${precio_fondo_proy:,.0f}")
if ya_en_fondo:
    print(f"  >>> YA PASAMOS EL FONDO PROYECTADO <<<")
else:
    print(f"  >>> Faltan ~{dias_al_fondo} dias para el fondo proyectado <<<")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.tick_params(colors="#c9d1d9", labelsize=9)
ax.spines[:].set_color("#30363d")

for s in series:
    lw = 2.2 if s["dias_fondo"] is None else 1.5
    ax.plot(s["dias"], s["idx"], color=s["color"], linewidth=lw,
            label=s["nombre"], zorder=3, alpha=0.9)
    # Pico
    ax.scatter(s["dias_pico"], s["idx_pico"],
               color=s["color"], s=70, zorder=6, edgecolors="#0d1117", lw=1.2)
    ax.annotate(f"Pico d+{s['dias_pico']}",
                xy=(s["dias_pico"], s["idx_pico"]),
                xytext=(6, 4), textcoords="offset points",
                color=s["color"], fontsize=7.5, zorder=7)
    # Fondo historico
    if s["dias_fondo"]:
        ax.scatter(s["dias_fondo"], s["idx_fondo"],
                   color=s["color"], s=90, marker="^", zorder=6,
                   edgecolors="#0d1117", lw=1.2)
        ax.annotate(f"Fondo d+{s['dias_fondo']}",
                    xy=(s["dias_fondo"], s["idx_fondo"]),
                    xytext=(6, -22), textcoords="offset points",
                    color=s["color"], fontsize=7.5, zorder=7)

# Linea de hoy
ax.axvline(dias_hoy, color="#ffffff", lw=1.2, ls="-.", alpha=0.6, zorder=4)
ax.text(dias_hoy + 6, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 1 else 500,
        f"HOY\nd+{dias_hoy}", color="#ffffff", fontsize=8, alpha=0.7,
        va="center", zorder=5)

# Fondo proyectado ciclo actual
ax.axvline(dias_fondo_proy, color="#f0b90b", lw=1.3, ls=":", alpha=0.85, zorder=4)
ax.scatter(dias_fondo_proy, idx_fondo_proy,
           color="#f0b90b", s=120, marker="^", zorder=6,
           edgecolors="#0d1117", lw=1.2)
ax.annotate(f"Fondo proy.\nd+{int(dias_fondo_proy)}\n~${precio_fondo_proy:,.0f}",
            xy=(dias_fondo_proy, idx_fondo_proy),
            xytext=(10, -40), textcoords="offset points",
            color="#f0b90b", fontsize=7.5, zorder=7,
            arrowprops=dict(arrowstyle="->", color="#f0b90b", lw=0.8))

ax.set_yscale("log")
ax.set_xlabel("Dias desde el halving", color="#c9d1d9", fontsize=11)
ax.set_ylabel("Precio indexado (halving = 100x)", color="#c9d1d9", fontsize=11)
ax.set_title("Bitcoin — Comparacion paralela de ciclos (indexado al halving)",
             color="#f0f6fc", fontsize=14, pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}x"))
ax.grid(True, color="#21262d", lw=0.5, zorder=0)
ax.set_xlim(0, max(c["ventana_dias"] for c in CICLOS))

# ── Panel de estado: comprar o esperar ───────────────────────────────────────
if ya_en_fondo:
    estado_color  = "#3fb950"
    estado_titulo = "MOMENTO DE COMPRA"
    estado_texto  = "El fondo proyectado ya paso.\nHistoricamente es zona de acumulacion."
    estado_icono  = "COMPRAR"
elif dias_al_fondo <= 90:
    estado_color  = "#f0b90b"
    estado_titulo = "ZONA DE ATENCION"
    estado_texto  = f"Faltan ~{dias_al_fondo} dias para el fondo.\nConsiderar acumulacion gradual."
    estado_icono  = "ACUMULAR"
else:
    estado_color  = "#f85149"
    estado_titulo = "ESPERAR"
    estado_texto  = f"Aun faltan ~{dias_al_fondo} dias para el fondo.\nHistoricamente hay mas caida."
    estado_icono  = "ESPERAR"

caida_restante = (precio_fondo_proy / precio_actual - 1) * 100
texto_panel = (
    f" {estado_icono}\n"
    f" {'─'*24}\n"
    f" Hoy: dia +{dias_hoy}\n"
    f" Precio actual:   ${precio_actual:>9,.0f}\n"
    f" Caida desde pico:{caida_actual:>+7.1f}%\n"
    f" Fondo proy.: {fecha_fondo_proy.strftime('%b %Y')} (d+{int(dias_fondo_proy)})\n"
    f" Precio fondo:    ${precio_fondo_proy:>9,.0f}\n"
    f" Caida restante: {caida_restante:>+7.1f}%"
)
# Cuadro pequeño en esquina inferior izquierda (zona vacia del grafico)
ax.text(0.013, 0.02, texto_panel,
        transform=ax.transAxes, va="bottom", ha="left",
        color="#c9d1d9", fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                  edgecolor=estado_color, linewidth=2, alpha=0.93),
        zorder=10)

# Leyenda
legend_items = [Line2D([0],[0], color=s["color"], lw=2, label=s["nombre"]) for s in series] + [
    Line2D([0],[0], color="#ffffff", lw=1.2, ls="-.", alpha=0.6, label=f"Hoy (d+{dias_hoy})"),
    Line2D([0],[0], color="#f0b90b", lw=1.3, ls=":", label="Fondo proyectado"),
]
ax.legend(handles=legend_items, facecolor="#161b22", edgecolor="#30363d",
          labelcolor="#c9d1d9", fontsize=8.5, loc="upper left")

fecha_export = datetime.today().strftime("%Y%m%d_%H%M")
nombre_archivo = f"bitcoin_ciclos_paralelos_{fecha_export}.png"
plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"\nGrafico guardado: {nombre_archivo}")
plt.show()
