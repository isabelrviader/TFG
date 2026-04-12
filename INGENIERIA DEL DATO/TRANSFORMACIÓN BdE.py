# -*- coding: utf-8 -*-
"""

TRANSFORMACIÓN — BLOQUE BdE (Banco de España)

"""
import statsmodels
print(statsmodels.__version__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH    = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\limpios'
OUTPUT_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\BDE'

FECHA_INICIO  = '2002-01-01'

FREQ_Q = 'QE'

os.makedirs(OUTPUT_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORES = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: VALIDACIÓN DEL DATASET
# Equivalente a df.info() + df.isnull().sum() + comprobación de duplicados.
# Se reutiliza en todos los pasos de validación.
# ──────────────────────────────────────────────────────────────────────────────

def validar_dataset(nombre, df):
    print(f"\n{'='*65}")
    print(f"  VALIDACION: {nombre}")
    print(f"{'='*65}")
    print(f"  Shape             : {df.shape}")
    print(f"  Rango temporal    : {df.index[0].date()}  ->  {df.index[-1].date()}")
    print(f"  Frecuencia        : trimestral")
    dup = df.index.duplicated().sum()
    print(f"  Fechas duplicadas : {dup}" + (" [OK]" if dup == 0 else " [REVISAR]"))
    print(f"\n  df.isnull().sum():")
    for col in df.columns:
        n   = df[col].isnull().sum()
        pct = round(n / len(df) * 100, 1)
        print(f"    · {col:<35} {n:>4} nulos ({pct:>5}%)")
    print(f"\n  df.head(3):")
    print(df.head(3).to_string())
    print(f"\n  df.tail(3):")
    print(df.tail(3).to_string())
    print(f"{'='*65}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: TEST ADF + KPSS (statsmodels)
#
# ADF  — H0: la serie tiene raiz unitaria (NO es estacionaria)
#         t-stat < valor critico 5% (-2.89) → rechazamos H0 → estacionaria
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

    # ADF (statsmodels): autolag='AIC' selecciona el numero optimo de lags
    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag='AIC')
    adf_ok = adf_stat < adf_cv['5%']

    # KPSS (statsmodels): nlags='auto', regression='c' (estacionaria en media)
    kpss_stat, kpss_p, _, kpss_cv = kpss(s, regression='c', nlags='auto')
    kpss_ok = kpss_stat < kpss_cv['5%']

    # Conclusion combinada
    if adf_ok and kpss_ok:
        conclusion = "Estacionaria"
        estac = True
    elif not adf_ok and kpss_ok:
        conclusion = "Prob. estacionaria"
        estac = True
    elif adf_ok and not kpss_ok:
        conclusion = "Dudosa (tendencia)"
        estac = False
    else:
        conclusion = "No estacionaria"
        estac = False

    return {
        'adf_stat':   adf_stat,
        'adf_p':      adf_p,
        'adf_cv5':    adf_cv['5%'],
        'adf_ok':     adf_ok,
        'kpss_stat':  kpss_stat,
        'kpss_p':     kpss_p,
        'kpss_cv5':   kpss_cv['5%'],
        'kpss_ok':    kpss_ok,
        'conclusion': conclusion,
        'estac':      estac,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.1.T — SELECCION DE VARIABLE OBJETIVO
# Carga los tres archivos limpios e identifica las variables objetivo.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 1.1.T — SELECCION DE VARIABLE OBJETIVO")
print("="*65)

be0413 = pd.read_excel(os.path.join(INPUT_PATH, "be0413_limpio.xlsx"),
                       index_col=0, parse_dates=True)
be1901 = pd.read_excel(os.path.join(INPUT_PATH, "be1901_limpio.xlsx"),
                       index_col=0, parse_dates=True)
be2507 = pd.read_excel(os.path.join(INPUT_PATH, "be2507_limpio.xlsx"),
                       index_col=0, parse_dates=True)

print(f"\n  Archivos cargados:")
print(f"    · be0413: {be0413.shape}  columnas: {list(be0413.columns)}")
print(f"    · be1901: {be1901.shape}  columnas: {list(be1901.columns)}")
print(f"    · be2507: {be2507.shape}  columnas: {list(be2507.columns)}")
print(f"\n  Variable objetivo : mora_hogares  (creditos dudosos hogares, miles EUR)")
print(f"  Variable referencia: mora_empresas (se mantiene pero no entra en modelo)")

# El corte temporal se fija en el minimo de los tres archivos para garantizar
# que todos los trimestres del dataset final tienen datos reales en las 3 fuentes.
ultimo_dato_real = min(be0413.index[-1], be1901.index[-1], be2507.index[-1])
print(f"\n  Ultimo dato real por fuente:")
print(f"    · be0413 : {be0413.index[-1].date()}")
print(f"    · be1901 : {be1901.index[-1].date()}")
print(f"    · be2507 : {be2507.index[-1].date()}")
print(f"  -> Corte aplicado (minimo): {ultimo_dato_real.date()}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.2.T — CONVERSION A SERIE TEMPORAL TRIMESTRAL
#
# be0413 y be2507: el BdE publica con frecuencia trimestral pero en indice
#   mensual. Los meses intermedios fueron interpolados en Exploracion y Limpieza.
#   resample('QE').last() → recupera el dato real del ultimo mes del trimestre.
#
# be1901 (Euribor): frecuencia mensual real.
#   resample('QE').mean() → media de los 3 meses del trimestre.
#   Tiene mas sentido economico porque el Euribor fluctua a lo largo del
#   trimestre y la media refleja mejor el coste medio de financiacion.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.2.T — CONVERSION A SERIE TEMPORAL TRIMESTRAL")
print("="*65)

be0413_q = be0413.resample(FREQ_Q).last()
be1901_q = be1901.resample(FREQ_Q).mean()
be2507_q = be2507.resample(FREQ_Q).last()

for nombre, df_m, df_q, metodo in [
    ("be0413", be0413, be0413_q, "last()"),
    ("be1901", be1901, be1901_q, "mean()"),
    ("be2507", be2507, be2507_q, "last()"),
]:
    print(f"\n  {nombre}  resample('{FREQ_Q}').{metodo}")
    print(f"    Antes  : {df_m.shape[0]} observaciones mensuales")
    print(f"    Despues: {df_q.shape[0]} observaciones trimestrales")
    print(f"    Rango  : {df_q.index[0].date()} -> {df_q.index[-1].date()}")

# Grafico G5: mensual vs trimestral (Euribor como ejemplo)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.suptitle("BdE — Efecto de la conversion a frecuencia trimestral\n"
             "(ejemplo: Euribor 12m mensual vs media trimestral)",
             fontsize=13, fontweight="bold")

axes[0].plot(be1901.index, be1901["euribor_12m"],
             color="#2196F3", linewidth=1.2, alpha=0.8, label="Mensual (original)")
axes[0].set_title("Euribor 12m — Serie mensual original", fontsize=11, fontweight="bold")
axes[0].set_ylabel("%")
axes[0].legend(fontsize=9)

axes[1].plot(be1901_q.index, be1901_q["euribor_12m"],
             color="#E91E63", linewidth=1.8, marker="o", markersize=3,
             label="Trimestral (media Q)")
axes[1].set_title("Euribor 12m — Serie trimestral (media)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("%")
axes[1].legend(fontsize=9)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G5_mensual_vs_trimestral_euribor.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G5 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.3.T — INTEGRACION DE LOS 3 DATASETS FINANCIEROS
#
# Se ejecuta ANTES de YoY y lags porque ambas transformaciones se calculan
# sobre el dataset unificado.
# Tras la union:
#   · Filtro desde 2002
#   · Eliminacion de trimestres incompletos (> ultimo dato real)
#   · Limpieza del indice (sin hora)
#   · Tratamiento de NaN residuales si los hubiera
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.5.T — INTEGRACION DE LOS 3 DATASETS FINANCIEROS")
print("="*65)

dataset_bde = pd.concat([be0413_q, be1901_q, be2507_q], axis=1, join='outer')
dataset_bde.index.name = 'fecha'

print(f"\n  Shape antes de filtrar  : {dataset_bde.shape}")
print(f"  Rango antes de filtrar  : {dataset_bde.index[0].date()} -> {dataset_bde.index[-1].date()}")

# Filtro temporal
dataset_bde = dataset_bde[dataset_bde.index >= FECHA_INICIO]
print(f"\n  Filtro temporal: desde {FECHA_INICIO}")
print(f"  Shape tras filtrar: {dataset_bde.shape}")

# Eliminar trimestres incompletos generados por resample cuando el archivo
# termina a mitad de trimestre
n_antes = len(dataset_bde)
dataset_bde = dataset_bde[dataset_bde.index <= ultimo_dato_real]
print(f"  Trimestres incompletos eliminados: {n_antes - len(dataset_bde)}")
print(f"  Ultimo trimestre real: {dataset_bde.index[-1].date()}")

# Eliminar hora del indice
dataset_bde.index = pd.DatetimeIndex(dataset_bde.index.date)
dataset_bde.index.name = 'fecha'

# NaN tras la union (por diferente cobertura temporal entre archivos)
nulos = dataset_bde.isnull().sum()
if nulos.sum() > 0:
    print(f"\n  [AVISO] NaN detectados tras la union:")
    for col, n in nulos[nulos > 0].items():
        print(f"    · {col}: {n} ({round(n/len(dataset_bde)*100,1)}%)")
    print("  -> Aplicando interpolacion temporal...")
    for col in dataset_bde.columns:
        if dataset_bde[col].isnull().sum() > 0:
            dataset_bde[col] = dataset_bde[col].interpolate(
                method='time', limit_direction='both')
    print(f"  NaN tras interpolacion: {dataset_bde.isnull().sum().sum()}")
else:
    print(f"\n  NaN tras la union: 0 (coberturas temporales alineadas)")

# Validacion con numpy ndarray
arr = dataset_bde.values
print(f"\n  Validacion ndarray:")
print(f"    · shape         : {arr.shape}")
print(f"    · np.isnan total: {np.isnan(arr).sum()}")
print(f"    · filas con NaN : {np.isnan(arr).any(axis=1).sum()}")

validar_dataset("dataset_BdE tras integracion", dataset_bde)

# Grafico G6: series temporales del dataset BdE integrado
variables_plot = {
    "credito_hogares":    ("Credito Hogares (miles EUR)",  "#2196F3"),
    "credito_empresas":   ("Credito Empresas (miles EUR)", "#E91E63"),
    "mora_hogares":       ("Mora Hogares (miles EUR)",      "#4CAF50"),
    "mora_empresas":      ("Mora Empresas (miles EUR)",     "#FF9800"),
    "euribor_12m":        ("Euribor 12m (%)",               "#9C27B0"),
    "precio_m2_vivienda": ("Precio m2 Vivienda (EUR/m2)",  "#00BCD4"),
}

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Dataset BdE — Series temporales trimestrales\n"
             "(dataset integrado y validado, desde 2002)",
             fontsize=13, fontweight="bold")

for ax, (col, (etiqueta, color)) in zip(axes.flat, variables_plot.items()):
    ax.plot(dataset_bde.index, dataset_bde[col],
            color=color, linewidth=1.8, marker="o", markersize=2)
    ax.set_title(etiqueta, fontsize=10, fontweight="bold")
    ax.set_ylabel(etiqueta.split("(")[-1].replace(")", ""), fontsize=8)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G6_series_temporales_bde_final.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G6 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.4.T — VARIABLES DERIVADAS YoY + ANALISIS DE ESTACIONARIEDAD
#
# YoY (Year-over-Year): tasa de variacion interanual.
# Formula: pct_change(4) → (valor_t / valor_t-4 - 1) * 100
# Con datos trimestrales, 4 periodos = 1 año.
#
# Variables transformadas a YoY:
#   · credito_hogares_yoy    → crecimiento interanual credito hogares
#   · credito_empresas_yoy   → crecimiento interanual credito empresas
#   · precio_m2_vivienda_yoy → crecimiento interanual precio vivienda
#
# NO se transforma a YoY:
#   · mora_hogares / mora_empresas: son variables objetivo, se usan en niveles
#   · euribor_12m: ya es un tipo en % directamente comparable entre periodos
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.3.T — VARIABLES DERIVADAS YoY + ESTACIONARIEDAD")
print("="*65)

# Calculo YoY
dataset_bde['credito_hogares_yoy']    = dataset_bde['credito_hogares'].pct_change(4) * 100
dataset_bde['credito_empresas_yoy']   = dataset_bde['credito_empresas'].pct_change(4) * 100
dataset_bde['precio_m2_vivienda_yoy'] = dataset_bde['precio_m2_vivienda'].pct_change(4) * 100

vars_yoy = ['credito_hogares_yoy', 'credito_empresas_yoy', 'precio_m2_vivienda_yoy']
print(f"\n  Variables YoY calculadas (pct_change(4) * 100):")
for col in vars_yoy:
    s = dataset_bde[col].dropna()
    print(f"    · {col:<30}  media: {s.mean():>6.2f}%  "
          f"min: {s.min():>7.2f}%  max: {s.max():>7.2f}%  "
          f"NaN: {dataset_bde[col].isna().sum()} (primeros 4 trimestres)")

# Analisis de estacionariedad ADF + KPSS (statsmodels)
# Se analiza cada variable en niveles y su version YoY para justificar
# la decision de transformar a YoY las variables de volumen/precio.
vars_estac = [
    ('mora_hogares',            'Nivel'),
    ('mora_empresas',           'Nivel'),
    ('credito_hogares',         'Nivel'),
    ('credito_empresas',        'Nivel'),
    ('euribor_12m',             'Nivel'),
    ('precio_m2_vivienda',      'Nivel'),
    ('credito_hogares_yoy',     'YoY'),
    ('credito_empresas_yoy',    'YoY'),
    ('precio_m2_vivienda_yoy',  'YoY'),
]

print(f"\n  Test ADF (statsmodels adfuller, autolag='AIC'):")
print(f"  H0: raiz unitaria (NO estacionaria). t < vc 5% → estacionaria")
print(f"\n  {'Variable':<35} {'t-stat':>8}  {'p-valor':>8}  {'vc 5%':>7}  {'Resultado'}")
print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")

resultados = {}
for col, tipo in vars_estac:
    r = test_estacionariedad(dataset_bde[col])
    resultados[col] = r
    res_adf = "Estac." if r['adf_ok'] else "No estac."
    print(f"  {col:<35} {r['adf_stat']:>8.3f}  {r['adf_p']:>8.4f}  "
          f"{r['adf_cv5']:>7.3f}  {res_adf}")

print(f"\n  Test KPSS (statsmodels kpss, regression='c', nlags='auto'):")
print(f"  H0: serie ES estacionaria. stat < vc 5% (0.463) → NO rechazamos H0")
print(f"\n  {'Variable':<35} {'stat':>8}  {'p-valor':>8}  {'vc 5%':>7}  {'Resultado'}")
print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")

for col, tipo in vars_estac:
    r = resultados[col]
    res_kpss = "Estac." if r['kpss_ok'] else "No estac."
    print(f"  {col:<35} {r['kpss_stat']:>8.4f}  {r['kpss_p']:>8.4f}  "
          f"{r['kpss_cv5']:>7.3f}  {res_kpss}")

print(f"\n  Conclusion combinada ADF + KPSS:")
print(f"  {'Variable':<35} {'ADF':>10}  {'KPSS':>10}  {'Conclusion'}")
print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*25}")
for col, tipo in vars_estac:
    r = resultados[col]
    print(f"  {col:<35} {'Si' if r['adf_ok'] else 'No':>10}  "
          f"{'Si' if r['kpss_ok'] else 'No':>10}  {r['conclusion']}")

# Grafico G7: tabla estadisticos + ADF + KPSS
vars_tabla = [col for col, _ in vars_estac]
filas = []
for col in vars_tabla:
    s = dataset_bde[col].dropna()
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
col_widths  = [0.22, 0.07, 0.08, 0.07, 0.07, 0.10, 0.10, 0.20]

fig, ax = plt.subplots(figsize=(20, 4.5))
ax.axis('off')
fig.suptitle("BdE — Estadisticos descriptivos + Test ADF + Test KPSS\n"
             "(analisis de estacionariedad combinado)",
             fontsize=13, fontweight="bold", y=1.02)

tabla = ax.table(
    cellText=filas,
    colLabels=col_headers,
    colWidths=col_widths,
    cellLoc='center',
    loc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(8.5)
tabla.scale(1, 1.7)

for j in range(len(col_headers)):
    tabla[0, j].set_facecolor('#212121')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

for i in range(len(filas)):
    for j in range(len(col_headers)):
        tabla[i+1, j].set_facecolor('#F5F5F5' if i % 2 == 0 else '#FFFFFF')
        tabla[i+1, j].set_text_props(color='black')
    tabla[i+1, j].set_linewidth(0.5)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G7_tabla_adf_kpss_bde.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G7 guardado: {ruta}")

# Grafico G8: variables YoY
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BdE — Variables derivadas YoY (variacion interanual)\n"
             "(credito hogares, credito empresas y precio vivienda)",
             fontsize=13, fontweight="bold")

yoy_plot = {
    'credito_hogares_yoy':    ("Credito Hogares YoY (%)",    "#2196F3"),
    'credito_empresas_yoy':   ("Credito Empresas YoY (%)",   "#E91E63"),
    'precio_m2_vivienda_yoy': ("Precio Vivienda YoY (%)",    "#4CAF50"),
}
for ax, (col, (etiqueta, color)) in zip(axes, yoy_plot.items()):
    s = dataset_bde[col].dropna()
    ax.plot(s.index, s.values, color=color, linewidth=1.8, marker="o", markersize=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(etiqueta, fontsize=10, fontweight="bold")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G8_variables_yoy_bde.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G8 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.5.T — VARIABLES REZAGADAS (LAGS)
#
# Se crean variables rezagadas para capturar efectos retardados del ciclo
# economico sobre la morosidad.
# Con datos trimestrales: lag=1 = 3 meses, lag=2 = 6 meses, lag=4 = 12 meses.
#
# Variables y lags seleccionados:
#   · euribor_12m: lags 1, 2, 4
#   · precio_m2_vivienda_yoy: lags 1, 2
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.4.T — VARIABLES REZAGADAS (LAGS)")
print("="*65)
print("  lag=1 -> 3 meses  |  lag=2 -> 6 meses  |  lag=4 -> 12 meses")

lags_config = {
    'euribor_12m':            [1, 2, 4],
    'precio_m2_vivienda_yoy': [1, 2],
}

for var, lags in lags_config.items():
    for lag in lags:
        col_lag = f"{var}_lag{lag}"
        dataset_bde[col_lag] = dataset_bde[var].shift(lag)
        print(f"  {col_lag:<40}  NaN introducidos: {dataset_bde[col_lag].isna().sum()}")

print(f"\n  Shape tras añadir lags: {dataset_bde.shape}")

# Grafico G9: Euribor original vs lags
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("BdE — Euribor 12m original vs variables rezagadas\n"
             "(lag 1 = 3 meses, lag 2 = 6 meses, lag 4 = 12 meses)",
             fontsize=13, fontweight="bold")

colores_lag   = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
etiquetas_lag = ["Original", "Lag 1 (3m)", "Lag 2 (6m)", "Lag 4 (12m)"]
series_lag    = ["euribor_12m", "euribor_12m_lag1", "euribor_12m_lag2", "euribor_12m_lag4"]

for col, color, etiqueta in zip(series_lag, colores_lag, etiquetas_lag):
    s = dataset_bde[col].dropna()
    ax.plot(s.index, s.values, color=color, linewidth=1.5, alpha=0.8, label=etiqueta)

ax.set_ylabel("%")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G9_euribor_lags.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G9 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR DATASET BdE FINAL
# El indice sin hora para que Excel muestre solo la fecha.
# ══════════════════════════════════════════════════════════════════════════════

ruta_output = os.path.join(OUTPUT_PATH, "dataset_BdE.xlsx")
dataset_bde_export = dataset_bde.copy()
dataset_bde_export.index = dataset_bde_export.index.strftime('%Y-%m-%d')
dataset_bde_export.index.name = 'fecha'
dataset_bde_export.to_excel(ruta_output)
print(f"\n  Dataset BdE guardado: {ruta_output}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — TRANSFORMACION BdE")
print("="*65)
print(f"\n  {'Variable':<30} {'Fuente':<10} {'Unidad':<15} {'Rol'}")
print(f"  {'-'*30} {'-'*10} {'-'*15} {'-'*20}")

resumen_vars = [
    ("mora_hogares",               "be0413", "miles EUR", "Variable objetivo"),
    ("mora_empresas",              "be0413", "miles EUR", "Referencia"),
    ("credito_hogares",            "be0413", "miles EUR", "Predictora"),
    ("credito_empresas",           "be0413", "miles EUR", "Predictora"),
    ("euribor_12m",                "be1901", "%",          "Predictora"),
    ("precio_m2_vivienda",         "be2507", "EUR/m2",    "Predictora"),
    ("credito_hogares_yoy",        "be0413", "%",          "Predictora YoY"),
    ("credito_empresas_yoy",       "be0413", "%",          "Predictora YoY"),
    ("precio_m2_vivienda_yoy",     "be2507", "%",          "Predictora YoY"),
    ("euribor_12m_lag1",           "be1901", "%",          "Lag 1 (3m)"),
    ("euribor_12m_lag2",           "be1901", "%",          "Lag 2 (6m)"),
    ("euribor_12m_lag4",           "be1901", "%",          "Lag 4 (12m)"),
    ("precio_m2_vivienda_yoy_lag1","be2507", "%",          "Lag 1 (3m)"),
    ("precio_m2_vivienda_yoy_lag2","be2507", "%",          "Lag 2 (6m)"),
]
for var, fuente, unidad, rol in resumen_vars:
    print(f"  {var:<30} {fuente:<10} {unidad:<15} {rol}")

print(f"\n  Shape final    : {dataset_bde.shape}")
print(f"  Rango temporal : {dataset_bde.index[0].date()} -> {dataset_bde.index[-1].date()}")
print(f"  Frecuencia     : trimestral")
print(f"  Total NaN      : {dataset_bde.isnull().sum().sum()}")
print(f"\n  Transformacion BdE completada.")
print(f"     Output   : {ruta_output}")
print(f"     Graficos :")
print(f"       G5  Mensual vs trimestral (Euribor)")
print(f"       G6  Series temporales dataset BdE final")
print(f"       G7  Tabla estadisticos + ADF + KPSS (b/n para memoria)")
print(f"       G8  Variables derivadas YoY")
print(f"       G9  Euribor original vs lags")