# -*- coding: utf-8 -*-
"""

TRANSFORMACIÓN — BLOQUE INE

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
TRANSF_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\INE'

FECHA_INICIO  = '2002-01-01'
FREQ_Q        = 'QE'

os.makedirs(TRANSF_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORES = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCION: VALIDACION DEL DATASET
# Equivalente a df.info() + df.isnull().sum() + comprobacion de duplicados.
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
# PASO 2.1.T — CONVERSION A FRECUENCIA TRIMESTRAL
#
# · 67198 (PIB) y 65079 (tasa_paro): ya son trimestrales → no necesitan resample
# · 50917 (IPC): frecuencia mensual real → resample('QE').mean()
#   Se usa la media trimestral
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 2.1.T — CONVERSION A FRECUENCIA TRIMESTRAL")
print("="*65)

ine_67198 = pd.read_excel(os.path.join(INPUT_PATH, "67198_limpio.xlsx"),
                          index_col=0, parse_dates=True)
ine_65079 = pd.read_excel(os.path.join(INPUT_PATH, "65079_limpio.xlsx"),
                          index_col=0, parse_dates=True)
ine_50917 = pd.read_excel(os.path.join(INPUT_PATH, "50917_limpio.xlsx"),
                          index_col=0, parse_dates=True)

print(f"\n  Archivos cargados:")
print(f"    · 67198 (PIB)       : {ine_67198.shape}  frecuencia: trimestral")
print(f"    · 65079 (Tasa Paro) : {ine_65079.shape}  frecuencia: trimestral")
print(f"    · 50917 (IPC)       : {ine_50917.shape}  frecuencia: mensual -> resample")

# PIB y tasa_paro ya son trimestrales
pib_q  = ine_67198.copy()
paro_q = ine_65079.copy()

# IPC: mensual -> media trimestral
print(f"\n  50917 (IPC mensual):")
print(f"  -> resample('{FREQ_Q}').mean()  [media trimestral]")
ipc_q = ine_50917.resample(FREQ_Q).mean()
print(f"  Antes  : {ine_50917.shape[0]} observaciones mensuales")
print(f"  Despues: {ipc_q.shape[0]} observaciones trimestrales")
print(f"  Rango  : {ipc_q.index[0].date()} -> {ipc_q.index[-1].date()}")

# Validacion con ndarray
print(f"\n  Validacion con numpy (ndarray):")
for nombre, df in [("PIB", pib_q), ("Tasa Paro", paro_q), ("IPC", ipc_q)]:
    arr = df.values
    print(f"    · {nombre:<12} shape: {arr.shape}  NaN: {np.isnan(arr).sum()}")

# Grafico G1: IPC mensual vs trimestral
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.suptitle("INE — Efecto de la conversion a frecuencia trimestral\n"
             "(ejemplo: IPC variacion anual mensual vs media trimestral)",
             fontsize=13, fontweight="bold")

axes[0].plot(ine_50917.index, ine_50917["ipc_var_anual"],
             color="#2196F3", linewidth=1.2, alpha=0.8, label="Mensual (original)")
axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[0].set_title("IPC Variacion Anual — Serie mensual original",
                  fontsize=11, fontweight="bold")
axes[0].set_ylabel("%")
axes[0].legend(fontsize=9)

axes[1].plot(ipc_q.index, ipc_q["ipc_var_anual"],
             color="#E91E63", linewidth=1.8, marker="o", markersize=3,
             label="Trimestral (media Q)")
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[1].set_title("IPC Variacion Anual — Serie trimestral (media)",
                  fontsize=11, fontweight="bold")
axes[1].set_ylabel("%")
axes[1].legend(fontsize=9)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G1_mensual_vs_trimestral_ipc.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G1 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.2.T — INTEGRACION DE DATASETS + FILTRO 2002
#
# Se unen los 3 datasets trimestrales 
# Tras la union:
#   · Filtro temporal desde 2002
#   · Eliminacion de trimestres incompletos (> minimo ultimo dato real)
#   · Limpieza del indice (sin hora)
#   · Tratamiento de NaN si los hubiera
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.2.T — INTEGRACION DE DATASETS + FILTRO 2002")
print("="*65)

# Fecha minima del ultimo dato real de las 3 fuentes
ultimo_dato_real = min(ine_67198.index[-1], ine_65079.index[-1], ipc_q.index[-1])

print(f"\n  Ultimo dato real por fuente:")
print(f"    · 67198 (PIB)       : {ine_67198.index[-1].date()}")
print(f"    · 65079 (Tasa Paro) : {ine_65079.index[-1].date()}")
print(f"    · 50917 (IPC)       : {ipc_q.index[-1].date()}")
print(f"  -> Corte aplicado (minimo): {ultimo_dato_real.date()}")

dataset_ine = pd.concat([pib_q, paro_q, ipc_q], axis=1, join='outer')
dataset_ine.index.name = 'fecha'

print(f"\n  Shape antes de filtrar  : {dataset_ine.shape}")

# Filtro temporal desde 2002
dataset_ine = dataset_ine[dataset_ine.index >= FECHA_INICIO]
print(f"  Filtro 2002 aplicado    : {dataset_ine.shape}")

# Eliminar trimestres incompletos generados por resample
n_antes = len(dataset_ine)
dataset_ine = dataset_ine[dataset_ine.index <= ultimo_dato_real]
print(f"  Trimestres eliminados   : {n_antes - len(dataset_ine)}")
print(f"  Ultimo trimestre real   : {dataset_ine.index[-1].date()}")

# Eliminar hora del indice
dataset_ine.index = pd.DatetimeIndex(dataset_ine.index.date)
dataset_ine.index.name = 'fecha'

# NaN tras la union
nulos = dataset_ine.isnull().sum()
if nulos.sum() > 0:
    print(f"\n  [AVISO] NaN detectados tras la union:")
    for col, n in nulos[nulos > 0].items():
        print(f"    · {col}: {n}")
    print("  -> Aplicando interpolacion temporal...")
    for col in dataset_ine.columns:
        if dataset_ine[col].isnull().sum() > 0:
            dataset_ine[col] = dataset_ine[col].interpolate(
                method='time', limit_direction='both')
    print(f"  NaN tras interpolacion: {dataset_ine.isnull().sum().sum()}")
else:
    print(f"\n  NaN tras la union: 0 (coberturas temporales alineadas)")

# Validacion con ndarray
arr = dataset_ine.values
print(f"\n  Validacion ndarray:")
print(f"    · shape         : {arr.shape}")
print(f"    · np.isnan total: {np.isnan(arr).sum()}")
print(f"    · filas con NaN : {np.isnan(arr).any(axis=1).sum()}")

validar_dataset("dataset_INE (integrado)", dataset_ine)

# Grafico G2: series temporales dataset INE integrado
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Dataset INE — Series temporales trimestrales\n"
             "(dataset integrado y validado, desde 2002)",
             fontsize=13, fontweight="bold")

vars_plot = [
    ("pib",           "PIB (indice vol. encadenado Base 2015)", "#2196F3"),
    ("tasa_paro",     "Tasa de Paro (%)",                       "#E91E63"),
    ("ipc_var_anual", "IPC Variacion Anual (%)",                "#4CAF50"),
]
for ax, (col, etiqueta, color) in zip(axes, vars_plot):
    ax.plot(dataset_ine.index, dataset_ine[col],
            color=color, linewidth=1.8, marker="o", markersize=2)
    if col in ["tasa_paro", "ipc_var_anual"]:
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(etiqueta, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G2_series_temporales_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G2 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.3.T — VARIABLES DERIVADAS YoY + ANALISIS DE ESTACIONARIEDAD
#
# Variables derivadas:
#   · pib_yoy: variacion interanual del PIB con pct_change(4)
#   · ipc_var_anual: ya es variacion anual — se usa directamente sin YoY.
#     Aplicar pct_change(4) sobre una variacion seria una doble diferencia
#     sin sentido economico.
#
# Analisis de estacionariedad: test ADF + KPSS (statsmodels) sobre todas
# las variables en niveles y sus versiones derivadas.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.3.T — VARIABLES DERIVADAS YoY + ESTACIONARIEDAD")
print("="*65)

# Calculo YoY del PIB
dataset_ine['pib_yoy'] = dataset_ine['pib'].pct_change(4) * 100

s = dataset_ine['pib_yoy'].dropna()
print(f"\n  pib_yoy (pct_change(4) * 100):")
print(f"    media: {s.mean():.2f}%  min: {s.min():.2f}%  "
      f"max: {s.max():.2f}%  NaN: {dataset_ine['pib_yoy'].isna().sum()} "
      f"(primeros 4 trimestres)")

# Grafico G3: PIB YoY e IPC variacion anual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("INE — Variables derivadas YoY\n"
             "(PIB variacion interanual e IPC variacion anual)",
             fontsize=13, fontweight="bold")

axes[0].plot(dataset_ine['pib_yoy'].dropna().index,
             dataset_ine['pib_yoy'].dropna().values,
             color="#2196F3", linewidth=1.8, marker="o", markersize=2)
axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[0].set_title("PIB YoY (%)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("%")
axes[0].tick_params(axis="x", rotation=30)

axes[1].plot(dataset_ine['ipc_var_anual'].dropna().index,
             dataset_ine['ipc_var_anual'].dropna().values,
             color="#E91E63", linewidth=1.8, marker="o", markersize=2)
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[1].set_title("IPC Variacion Anual (%, directo)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("%")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G3_variables_yoy_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G3 guardado: {ruta}")

# Analisis de estacionariedad ADF + KPSS
vars_estac = [
    ('pib',           'Nivel'),
    ('tasa_paro',     'Nivel'),
    ('ipc_var_anual', 'Nivel/Var'),
    ('pib_yoy',       'YoY'),
]

print(f"\n  Test ADF (statsmodels adfuller, autolag='AIC'):")
print(f"  H0: raiz unitaria (NO estacionaria). t < vc 5% -> estacionaria")
print(f"\n  {'Variable':<30} {'t-stat':>8}  {'p-valor':>8}  {'vc 5%':>7}  {'Resultado'}")
print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*15}")

resultados = {}
for col, tipo in vars_estac:
    r = test_estacionariedad(dataset_ine[col])
    resultados[col] = r
    print(f"  {col:<30} {r['adf_stat']:>8.3f}  {r['adf_p']:>8.4f}  "
          f"{r['adf_cv5']:>7.3f}  {'Estac.' if r['adf_ok'] else 'No estac.'}")

print(f"\n  Test KPSS (statsmodels kpss, regression='c', nlags='auto'):")
print(f"  H0: serie ES estacionaria. stat < vc 5% (0.463) -> NO rechazamos H0")
print(f"\n  {'Variable':<30} {'stat':>8}  {'p-valor':>8}  {'vc 5%':>7}  {'Resultado'}")
print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*15}")

for col, tipo in vars_estac:
    r = resultados[col]
    print(f"  {col:<30} {r['kpss_stat']:>8.4f}  {r['kpss_p']:>8.4f}  "
          f"{r['kpss_cv5']:>7.3f}  {'Estac.' if r['kpss_ok'] else 'No estac.'}")

print(f"\n  Conclusion combinada ADF + KPSS:")
print(f"  {'Variable':<30} {'ADF':>10}  {'KPSS':>10}  {'Conclusion'}")
print(f"  {'-'*30} {'-'*10}  {'-'*10}  {'-'*20}")
for col, tipo in vars_estac:
    r = resultados[col]
    print(f"  {col:<30} {'Si' if r['adf_ok'] else 'No':>10}  "
          f"{'Si' if r['kpss_ok'] else 'No':>10}  {r['conclusion']}")

# Grafico G4: tabla estadisticos + ADF + KPSS
vars_tabla = [col for col, _ in vars_estac]
filas = []
for col in vars_tabla:
    s = dataset_ine[col].dropna()
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
col_widths  = [0.20, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.24]

fig, ax = plt.subplots(figsize=(18, 3.5))
ax.axis('off')
fig.suptitle("INE — Estadisticos descriptivos + Test ADF + Test KPSS\n"
             "(analisis de estacionariedad combinado)",
             fontsize=13, fontweight="bold", y=1.05)

tabla = ax.table(
    cellText=filas,
    colLabels=col_headers,
    colWidths=col_widths,
    cellLoc='center',
    loc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(8.5)
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
ruta = os.path.join(GRAFICOS_PATH, "G4_tabla_adf_kpss_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G4 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.4.T — CARGA DATASET BdE + RATIO CREDITO TOTAL / PIB
#
# Se carga el dataset_BdE.xlsx ya transformado para calcular:
#   ratio_credito_pib = (credito_hogares + credito_empresas) / pib_nominal * 100
#
# El PIB del INE esta en indice Base 2015=100, no en EUR.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.4.T — CARGA DATASET BdE + RATIO CREDITO TOTAL / PIB")
print("="*65)

dataset_bde = pd.read_excel(os.path.join(TRANSF_PATH, "dataset_BdE.xlsx"),
                             index_col=0, parse_dates=True)
dataset_bde.index = pd.DatetimeIndex(dataset_bde.index.date)
dataset_bde.index.name = 'fecha'

print(f"  dataset_BdE cargado: {dataset_bde.shape}")
print(f"  Columnas: {list(dataset_bde.columns[:6])}")

# Alinear indices BdE e INE al periodo comun
idx_comun = dataset_ine.index.intersection(dataset_bde.index)
print(f"\n  Periodo comun BdE e INE: {idx_comun[0].date()} -> {idx_comun[-1].date()}")
print(f"  Trimestres comunes: {len(idx_comun)}")

bde_alin = dataset_bde.loc[idx_comun]
ine_alin = dataset_ine.loc[idx_comun]

# Credito total en miles EUR
credito_total = bde_alin['credito_hogares'] + bde_alin['credito_empresas']

# PIB nominal aproximado: indice * PIB nominal 2015
PIB_NOMINAL_2015_MILES = 1_077_000_000
pib_nominal_miles = (ine_alin['pib'] / 100) * PIB_NOMINAL_2015_MILES

# Ratio credito total / PIB en %
ratio_credito_pib = (credito_total / pib_nominal_miles) * 100

print(f"\n  Ratio Credito Total / PIB:")
print(f"    media : {ratio_credito_pib.mean():.1f}%")
print(f"    min   : {ratio_credito_pib.min():.1f}%  "
      f"(fecha: {ratio_credito_pib.idxmin().date()})")
print(f"    max   : {ratio_credito_pib.max():.1f}%  "
      f"(fecha: {ratio_credito_pib.idxmax().date()})")

# Grafico G5: ratio credito/PIB
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ratio_credito_pib.index, ratio_credito_pib.values,
        color="#FF9800", linewidth=2, marker="o", markersize=2)
ax.fill_between(ratio_credito_pib.index, ratio_credito_pib.values,
                alpha=0.15, color="#FF9800")
ax.set_title("INE+BdE — Ratio Credito Total / PIB (%)\n"
             "(indicador de endeudamiento de la economia española)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("% sobre PIB")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G5_ratio_credito_pib.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G5 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.5.T — VARIABLES REZAGADAS (LAGS)
#
# Variables y lags seleccionados:
#   · tasa_paro: lags 1, 2, 4
#   · pib_yoy: lags 1, 2
#   · ipc_var_anual: lag 1
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.5.T — VARIABLES REZAGADAS (LAGS)")
print("="*65)
print("  lag=1 -> 3 meses  |  lag=2 -> 6 meses  |  lag=4 -> 12 meses")

# Aplicar lags sobre el dataset alineado al periodo comun
dataset_ine_alin = ine_alin.copy()
dataset_ine_alin['pib_yoy'] = dataset_ine['pib_yoy'].reindex(idx_comun)

lags_config = {
    'tasa_paro':     [1, 2, 4],
    'pib_yoy':       [1, 2],
    'ipc_var_anual': [1],
}

for var, lags in lags_config.items():
    for lag in lags:
        col_lag = f"{var}_lag{lag}"
        dataset_ine_alin[col_lag] = dataset_ine_alin[var].shift(lag)
        print(f"  {col_lag:<35}  NaN introducidos: "
              f"{dataset_ine_alin[col_lag].isna().sum()}")

print(f"\n  Shape tras añadir lags: {dataset_ine_alin.shape}")

# Grafico G6: tasa paro original vs lags
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("INE — Tasa de Paro original vs variables rezagadas\n"
             "(lag 1 = 3 meses, lag 2 = 6 meses, lag 4 = 12 meses)",
             fontsize=13, fontweight="bold")

colores_lag   = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
etiquetas_lag = ["Original", "Lag 1 (3m)", "Lag 2 (6m)", "Lag 4 (12m)"]
series_lag    = ["tasa_paro", "tasa_paro_lag1", "tasa_paro_lag2", "tasa_paro_lag4"]

for col, color, etiqueta in zip(series_lag, colores_lag, etiquetas_lag):
    s = dataset_ine_alin[col].dropna()
    ax.plot(s.index, s.values, color=color, linewidth=1.5,
            alpha=0.8, label=etiqueta)

ax.set_ylabel("%")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G6_tasa_paro_lags.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G6 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.6.T — UNION DATASET INE + RATIO → DATASET INE FINAL
#
# Se añade el ratio credito/PIB al dataset INE y se valida el resultado final.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.6.T — UNION DATASET INE + RATIO → DATASET INE FINAL")
print("="*65)

dataset_ine_final = dataset_ine_alin.copy()
dataset_ine_final['ratio_credito_pib'] = ratio_credito_pib

# Validacion con ndarray
arr = dataset_ine_final.values
print(f"\n  Validacion ndarray:")
print(f"    · shape         : {arr.shape}")
print(f"    · np.isnan total: {np.isnan(arr).sum()}")
print(f"    · filas con NaN : {np.isnan(arr).any(axis=1).sum()}")
print(f"      (NaN esperados por lags y pib_yoy — primeros trimestres)")

validar_dataset("dataset_INE FINAL", dataset_ine_final)

# Guardar con indice sin hora
ruta_output = os.path.join(TRANSF_PATH, "dataset_INE.xlsx")
dataset_ine_export = dataset_ine_final.copy()
dataset_ine_export.index = dataset_ine_export.index.strftime('%Y-%m-%d')
dataset_ine_export.index.name = 'fecha'
dataset_ine_export.to_excel(ruta_output)
print(f"\n  Dataset INE guardado: {ruta_output}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — TRANSFORMACION INE")
print("="*65)
print(f"\n  {'Variable':<30} {'Fuente':<10} {'Unidad':<20} {'Tipo'}")
print(f"  {'-'*30} {'-'*10} {'-'*20} {'-'*20}")

resumen_vars = [
    ("pib",                "67198",   "Indice B2015",  "Predictora base"),
    ("tasa_paro",          "65079",   "%",             "Predictora base"),
    ("ipc_var_anual",      "50917",   "%",             "Predictora base"),
    ("pib_yoy",            "67198",   "%",             "Derivada YoY"),
    ("tasa_paro_lag1",     "65079",   "%",             "Lag 1 (3m)"),
    ("tasa_paro_lag2",     "65079",   "%",             "Lag 2 (6m)"),
    ("tasa_paro_lag4",     "65079",   "%",             "Lag 4 (12m)"),
    ("pib_yoy_lag1",       "67198",   "%",             "Lag 1 (3m)"),
    ("pib_yoy_lag2",       "67198",   "%",             "Lag 2 (6m)"),
    ("ipc_var_anual_lag1", "50917",   "%",             "Lag 1 (3m)"),
    ("ratio_credito_pib",  "BdE+INE", "% PIB",         "Indicador financiero"),
]
for var, fuente, unidad, tipo in resumen_vars:
    print(f"  {var:<30} {fuente:<10} {unidad:<20} {tipo}")

print(f"\n  Shape final    : {dataset_ine_final.shape}")
print(f"  Rango temporal : {dataset_ine_final.index[0].date()} -> "
      f"{dataset_ine_final.index[-1].date()}")
print(f"  Frecuencia     : trimestral")
print(f"  Total NaN      : {dataset_ine_final.isnull().sum().sum()}")
print(f"\n  Transformacion INE completada.")
print(f"     Output   : {ruta_output}")
print(f"     Graficos :")
print(f"       G1  IPC mensual vs trimestral")
print(f"       G2  Series temporales dataset INE integrado")
print(f"       G3  PIB YoY e IPC variacion anual")
print(f"       G4  Tabla estadisticos + ADF + KPSS (b/n para memoria)")
print(f"       G5  Ratio Credito Total / PIB")
print(f"       G6  Tasa Paro original vs lags")