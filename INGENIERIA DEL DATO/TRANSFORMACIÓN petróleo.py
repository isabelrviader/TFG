# -*- coding: utf-8 -*-
"""

TRANSFORMACIÓN — BLOQUE PETRÓLEO (Brent Crude Oil)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH    = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\limpios'
OUTPUT_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\PETROLEO'

FECHA_INICIO = '2002-01-01'
FREQ_Q       = 'QE'
COLOR_BRENT  = "#FF9800"

os.makedirs(OUTPUT_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCION: TEST ADF + KPSS (statsmodels)
#
# ADF  — H0: la serie tiene raiz unitaria (NO es estacionaria)
#         t-stat < valor critico 5% → rechazamos H0 → estacionaria
#
# KPSS — H0: la serie ES estacionaria
#         estadistico < valor critico 5% (0.463) → NO rechazamos H0 → estacionaria
#
# Conclusion combinada:
#   ADF si + KPSS si → estacionaria
#   ADF no + KPSS si → probablemente estacionaria (ADF poco potente)
#   ADF si + KPSS no → estacionaria en tendencia (dudosa)
#   ADF no + KPSS no → no estacionaria
# ──────────────────────────────────────────────────────────────────────────────

def test_estacionariedad(serie):
    """
    Aplica ADF y KPSS sobre una serie sin NaN.
    Devuelve dict con estadisticos, p-valores y conclusion combinada.
    """
    s = serie.dropna()

    # ADF: autolag='AIC' selecciona el numero optimo de lags automaticamente
    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag='AIC')
    adf_ok = adf_stat < adf_cv['5%']

    # KPSS: nlags='auto', regression='c' (estacionaria en media)
    kpss_stat, kpss_p, _, kpss_cv = kpss(s, regression='c', nlags='auto')
    kpss_ok = kpss_stat < kpss_cv['5%']

    if adf_ok and kpss_ok:
        conclusion, estac = "Estacionaria", True
    elif not adf_ok and kpss_ok:
        conclusion, estac = "Prob. estacionaria", True
    elif adf_ok and not kpss_ok:
        conclusion, estac = "Dudosa (tendencia)", False
    else:
        conclusion, estac = "No estacionaria", False

    return {
        'adf_stat':  adf_stat,
        'adf_p':     adf_p,
        'adf_cv5':   adf_cv['5%'],
        'adf_ok':    adf_ok,
        'kpss_stat': kpss_stat,
        'kpss_p':    kpss_p,
        'kpss_cv5':  kpss_cv['5%'],
        'kpss_ok':   kpss_ok,
        'conclusion': conclusion,
        'estac':     estac,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3.1.T — CONVERSION TRIMESTRAL + FILTRO 2002 + ELIMINAR HORA
#
# El Brent es mensual real → resample('QE').mean()
# Se usa la media trimestral
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 3.1.T — CONVERSION TRIMESTRAL + FILTRO 2002 + HORA")
print("="*65)

df = pd.read_excel(os.path.join(INPUT_PATH, "petroleo_limpio.xlsx"),
                   index_col=0, parse_dates=True)
print(f"\n  Archivo cargado: {df.shape}  columnas: {list(df.columns)}")
print(f"  Rango original : {df.index[0].date()} -> {df.index[-1].date()}")

# Conversion mensual → trimestral con media
brent_q = df.resample(FREQ_Q).mean()
print(f"\n  Resample('{FREQ_Q}').mean():")
print(f"    Antes  : {df.shape[0]} observaciones mensuales")
print(f"    Despues: {brent_q.shape[0]} observaciones trimestrales")

# Filtro desde 2002
brent_q = brent_q[brent_q.index >= FECHA_INICIO]
print(f"\n  Filtro desde {FECHA_INICIO}: {brent_q.shape[0]} trimestres")

# Eliminar trimestres incompletos — el resample puede generar un trimestre
# final con valores repetidos si el archivo termina a mitad de trimestre
ultimo_dato_real = df.index[-1]
n_antes = len(brent_q)
brent_q = brent_q[brent_q.index <= ultimo_dato_real]
print(f"  Trimestres incompletos eliminados: {n_antes - len(brent_q)}")
print(f"  Ultimo trimestre real: {brent_q.index[-1].date()}")

# Eliminar hora del indice
brent_q.index = pd.DatetimeIndex(brent_q.index.date)
brent_q.index.name = 'fecha'

# Validacion con ndarray
arr = brent_q.values
print(f"\n  Validacion ndarray:")
print(f"    · shape     : {arr.shape}")
print(f"    · NaN total : {np.isnan(arr).sum()}")

# Grafico G1: Brent mensual vs trimestral (desde 2002 para comparar el mismo periodo)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.suptitle("Petroleo Brent — Efecto de la conversion a frecuencia trimestral\n"
             "(serie mensual original vs media trimestral, desde 2002)",
             fontsize=13, fontweight="bold")

df_2002 = df[df.index >= FECHA_INICIO]
axes[0].plot(df_2002.index, df_2002['brent'],
             color=COLOR_BRENT, linewidth=1.2, alpha=0.8, label="Mensual (original)")
axes[0].set_title("Brent — Serie mensual original (desde 2002)",
                  fontsize=11, fontweight="bold")
axes[0].set_ylabel("$/bbl")
axes[0].legend(fontsize=9)

axes[1].plot(brent_q.index, brent_q['brent'],
             color="#E65100", linewidth=1.8, marker="o", markersize=3,
             label="Trimestral (media Q)")
axes[1].set_title("Brent — Serie trimestral (media)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("$/bbl")
axes[1].legend(fontsize=9)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G5_mensual_vs_trimestral_brent.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G5 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3.2.T — BRENT YoY + ANALISIS DE ESTACIONARIEDAD ADF + KPSS
#
# Se calcula brent_yoy con pct_change(4) — 4 trimestres = 1 año.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 3.2.T — BRENT YoY + ANALISIS DE ESTACIONARIEDAD")
print("="*65)

# Calculo YoY
brent_q['brent_yoy'] = brent_q['brent'].pct_change(4) * 100
s_yoy = brent_q['brent_yoy'].dropna()

print(f"\n  brent_yoy = brent.pct_change(4) * 100:")
print(f"    media : {s_yoy.mean():.2f}%")
print(f"    min   : {s_yoy.min():.2f}%  (fecha: {s_yoy.idxmin().date()})")
print(f"    max   : {s_yoy.max():.2f}%  (fecha: {s_yoy.idxmax().date()})")
print(f"    NaN   : {brent_q['brent_yoy'].isna().sum()} (primeros 4 trimestres)")

# Grafico G2: serie trimestral + YoY
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Petroleo Brent — Serie trimestral y variacion interanual YoY",
             fontsize=13, fontweight="bold")

axes[0].plot(brent_q.index, brent_q['brent'],
             color=COLOR_BRENT, linewidth=1.8, marker="o", markersize=2)
axes[0].set_title("Brent trimestral ($/bbl)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("$/bbl")
axes[0].tick_params(axis="x", rotation=30)

axes[1].plot(s_yoy.index, s_yoy.values,
             color="#E65100", linewidth=1.8, marker="o", markersize=2)
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[1].set_title("Brent YoY (%)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("%")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G6_brent_trimestral_yoy.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G6 guardado: {ruta}")

# Analisis de estacionariedad ADF + KPSS (statsmodels)
vars_test = [('brent', 'Nivel'), ('brent_yoy', 'YoY')]
resultados = {}

print(f"\n  Test ADF (statsmodels adfuller, autolag='AIC'):")
print(f"  H0: raiz unitaria (NO estacionaria). t < vc 5% -> estacionaria")
print(f"\n  {'Variable':<20} {'t-stat':>8}  {'p-valor':>8}  {'vc 5%':>7}  {'Resultado'}")
print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*15}")

for col, tipo in vars_test:
    r = test_estacionariedad(brent_q[col])
    resultados[col] = r
    print(f"  {col:<20} {r['adf_stat']:>8.3f}  {r['adf_p']:>8.4f}  "
          f"{r['adf_cv5']:>7.3f}  {'Estac.' if r['adf_ok'] else 'No estac.'}")

print(f"\n  Test KPSS (statsmodels kpss, regression='c', nlags='auto'):")
print(f"  H0: serie ES estacionaria. stat < vc 5% (0.463) -> NO rechazamos H0")
print(f"\n  {'Variable':<20} {'stat':>8}  {'p-valor':>8}  {'vc 5%':>7}  {'Resultado'}")
print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*15}")

for col, tipo in vars_test:
    r = resultados[col]
    print(f"  {col:<20} {r['kpss_stat']:>8.4f}  {r['kpss_p']:>8.4f}  "
          f"{r['kpss_cv5']:>7.3f}  {'Estac.' if r['kpss_ok'] else 'No estac.'}")

print(f"\n  Conclusion combinada ADF + KPSS:")
print(f"  {'Variable':<20} {'ADF':>10}  {'KPSS':>10}  {'Conclusion'}")
print(f"  {'-'*20} {'-'*10}  {'-'*10}  {'-'*20}")
for col, tipo in vars_test:
    r = resultados[col]
    print(f"  {col:<20} {'Si' if r['adf_ok'] else 'No':>10}  "
          f"{'Si' if r['kpss_ok'] else 'No':>10}  {r['conclusion']}")

# Grafico G3: tabla estadisticos + ADF + KPSS
filas = []
for col, _ in vars_test:
    s = brent_q[col].dropna()
    r = resultados[col]
    filas.append([
        col,
        f"{s.mean():.2f}",
        f"{s.std():.2f}",
        f"{s.min():.2f}",
        f"{s.max():.2f}",
        f"{r['adf_stat']:.3f}",
        f"{r['kpss_stat']:.4f}",
        r['conclusion'],
    ])

col_headers = ['Variable', 'Media', 'Desv.Tip.', 'Min', 'Max',
               't-stat ADF', 'KPSS-stat', 'Conclusion ADF+KPSS']
col_widths  = [0.16, 0.08, 0.08, 0.08, 0.08, 0.11, 0.11, 0.26]

fig, ax = plt.subplots(figsize=(17, 2.8))
ax.axis('off')
fig.suptitle("Petroleo — Estadisticos descriptivos + Test ADF + Test KPSS\n"
             "(analisis de estacionariedad combinado)",
             fontsize=13, fontweight="bold", y=1.05)

tabla = ax.table(cellText=filas, colLabels=col_headers,
                 colWidths=col_widths, cellLoc='center', loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 1.8)

for j in range(len(col_headers)):
    tabla[0, j].set_facecolor('#212121')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

for i in range(len(filas)):
    for j in range(len(col_headers)):
        tabla[i+1, j].set_facecolor('#F5F5F5' if i % 2 == 0 else '#FFFFFF')
        tabla[i+1, j].set_text_props(color='black')
        tabla[i+1, j].set_linewidth(0.5)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G7_tabla_adf_kpss_brent.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G7 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3.3.T — VARIABLES REZAGADAS (LAGS)
#
# Se crean lags de brent_yoy para capturar el efecto retardado del
# precio del petroleo sobre la economia:
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 3.3.T — VARIABLES REZAGADAS (LAGS)")
print("="*65)
print("  lag=1 -> 3 meses  |  lag=2 -> 6 meses  |  lag=4 -> 12 meses")

for lag in [1, 2, 4]:
    col_lag = f"brent_yoy_lag{lag}"
    brent_q[col_lag] = brent_q['brent_yoy'].shift(lag)
    print(f"  {col_lag:<25}  NaN introducidos: {brent_q[col_lag].isna().sum()}")

print(f"\n  Shape tras añadir lags: {brent_q.shape}")

# Grafico G4: brent_yoy original vs lags
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Petroleo Brent — YoY original vs variables rezagadas\n"
             "(lag 1 = 3 meses, lag 2 = 6 meses, lag 4 = 12 meses)",
             fontsize=13, fontweight="bold")

colores_lag   = [COLOR_BRENT, "#E65100", "#BF360C", "#4E342E"]
etiquetas_lag = ["YoY original", "Lag 1 (3m)", "Lag 2 (6m)", "Lag 4 (12m)"]
series_lag    = ["brent_yoy", "brent_yoy_lag1", "brent_yoy_lag2", "brent_yoy_lag4"]

for col, color, etiqueta in zip(series_lag, colores_lag, etiquetas_lag):
    s = brent_q[col].dropna()
    ax.plot(s.index, s.values, color=color, linewidth=1.5,
            alpha=0.8, label=etiqueta)

ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_ylabel("%")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G8_brent_yoy_lags.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G8 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR DATASET PETROLEO FINAL
# El indice se convierte a string sin hora para que Excel muestre solo la fecha.
# ══════════════════════════════════════════════════════════════════════════════

ruta_output = os.path.join(OUTPUT_PATH, "dataset_petroleo.xlsx")
df_export = brent_q.copy()
df_export.index = df_export.index.strftime('%Y-%m-%d')
df_export.index.name = 'fecha'
df_export.to_excel(ruta_output)
print(f"\n  dataset_petroleo.xlsx guardado: {ruta_output}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — TRANSFORMACION PETROLEO")
print("="*65)
print(f"\n  {'Variable':<25} {'Unidad':<10} {'Tipo'}")
print(f"  {'-'*25} {'-'*10} {'-'*20}")

resumen = [
    ("brent",          "$/bbl", "Predictora base"),
    ("brent_yoy",      "%",     "Derivada YoY"),
    ("brent_yoy_lag1", "%",     "Lag 1 (3m)"),
    ("brent_yoy_lag2", "%",     "Lag 2 (6m)"),
    ("brent_yoy_lag4", "%",     "Lag 4 (12m)"),
]
for var, unidad, tipo in resumen:
    print(f"  {var:<25} {unidad:<10} {tipo}")

print(f"\n  Shape final    : {brent_q.shape}")
print(f"  Rango temporal : {brent_q.index[0].date()} -> {brent_q.index[-1].date()}")
print(f"  Frecuencia     : trimestral")
print(f"  Total NaN      : {brent_q.isnull().sum().sum()}")
print(f"\n  Transformacion petroleo completada.")
print(f"     Output   : {ruta_output}")
print(f"     Graficos :")
print(f"       G5  Brent mensual vs trimestral")
print(f"       G6  Brent trimestral + YoY")
print(f"       G7  Tabla estadisticos + ADF + KPSS (b/n para memoria)")
print(f"       G8  Brent YoY original vs lags")