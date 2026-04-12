# -*- coding: utf-8 -*-
"""

INTEGRACIÓN FINAL — DATASET MACRO-FINANCIERO

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

TRANSF_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados'
TRANSF_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\FINAL'

os.makedirs(TRANSF_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORES = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: RESUMEN DATASET
# Imprime nº observaciones, variables, dtype, % nulos y rango temporal.
# Se usa antes y después de la limpieza para comparar el estado del dataset.
# ──────────────────────────────────────────────────────────────────────────────

def resumen_dataset(nombre, df):
    print(f"\n{'='*65}")
    print(f"  DATASET: {nombre}")
    print(f"{'='*65}")
    print(f"  Shape             : {df.shape}")
    print(f"  Observaciones     : {df.shape[0]}")
    print(f"  Variables         : {df.shape[1]}")
    print(f"  Rango temporal    : {df.index[0]}  →  {df.index[-1]}")
    print(f"  Fechas duplicadas : {df.index.duplicated().sum()}"
          if df.index.duplicated().sum() == 0
          else f"  Fechas duplicadas: {df.index.duplicated().sum()}")
    print(f"  Tipos de dato     : {dict(df.dtypes.value_counts())}")
    print(f"\n  df.isna().sum():")
    total_nulos = 0
    for col in df.columns:
        n   = df[col].isnull().sum()
        pct = round(n / len(df) * 100, 1)
        estado = "OK" if n == 0 else "No OK"
        print(f"    · {col:<35} {n:>4} nulos ({pct:>5}%)  {estado}")
        total_nulos += n
    print(f"\n  Total NaN         : {total_nulos}")
    print(f"  Completitud global: {round((1 - total_nulos/(df.shape[0]*df.shape[1]))*100,2)}%")
    print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4.1 — CARGA Y REVISIÓN DE LOS 3 DATASETS TRANSFORMADOS
#
# Se cargan los 3 datasets limpios y transformados.
# Se revisa el estado inicial de cada uno antes de la integración:
# shape, columnas, rango temporal, nulos y tipos de dato.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 4.1 — CARGA Y REVISIÓN DE LOS 3 DATASETS")
print("="*65)

# Carga de los 3 datasets transformados
# Se elimina la hora
df_bde = pd.read_excel(os.path.join(TRANSF_PATH, "dataset_BdE.xlsx"),
                       index_col=0, parse_dates=True)
df_ine = pd.read_excel(os.path.join(TRANSF_PATH, "dataset_INE.xlsx"),
                       index_col=0, parse_dates=True)
df_pet = pd.read_excel(os.path.join(TRANSF_PATH, "dataset_petroleo.xlsx"),
                       index_col=0, parse_dates=True)

# Eliminar hora del índice en los 3 datasets
for df in [df_bde, df_ine, df_pet]:
    df.index = pd.DatetimeIndex(df.index.date)
    df.index.name = 'fecha'

print(f"\n  Datasets cargados:")
print(f"    · dataset_BdE     : {df_bde.shape}  "
      f"rango: {df_bde.index[0]} → {df_bde.index[-1]}")
print(f"    · dataset_INE     : {df_ine.shape}  "
      f"rango: {df_ine.index[0]} → {df_ine.index[-1]}")
print(f"    · dataset_petroleo: {df_pet.shape}  "
      f"rango: {df_pet.index[0]} → {df_pet.index[-1]}")

print(f"\n  Columnas por dataset:")
print(f"    · BdE     : {list(df_bde.columns)}")
print(f"    · INE     : {list(df_ine.columns)}")
print(f"    · Petróleo: {list(df_pet.columns)}")

# Verificar periodo común entre los 3 datasets
fecha_inicio = max(df_bde.index[0], df_ine.index[0], df_pet.index[0])
fecha_fin    = min(df_bde.index[-1], df_ine.index[-1], df_pet.index[-1])
print(f"\n  Periodo común de los 3 datasets:")
print(f"    Inicio: {fecha_inicio}  →  Fin: {fecha_fin}")

# df.head() y df.tail() de cada dataset para revisión visual
for nombre, df in [("BdE", df_bde), ("INE", df_ine), ("Petróleo", df_pet)]:
    print(f"\n  >>> {nombre} — df.head(2):")
    print(df.head(2).to_string())
    print(f"\n  >>> {nombre} — df.tail(2):")
    print(df.tail(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4.2 — INTEGRACIÓN → DATASET MACRO-FINANCIERO
#
# Se unen los 3 datasets con pd.concat(axis=1, join='outer') para no
# perder ningún periodo y detectar posibles desajustes temporales.
# Se revisa el estado inmediatamente después de la unión.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 4.2 — INTEGRACIÓN → DATASET MACRO-FINANCIERO")
print("="*65)
print("  Método: pd.concat([df_bde, df_ine, df_pet], axis=1, join='outer')")

df_final = pd.concat([df_bde, df_ine, df_pet], axis=1, join='outer')
df_final.index.name = 'fecha'

print(f"\n  ── ESTADO TRAS LA UNIÓN (antes de limpieza) ──────────────")
print(f"  Shape                : {df_final.shape}")
print(f"  Nº variables         : {df_final.shape[1]}")
print(f"  Nº observaciones     : {df_final.shape[0]}")
print(f"  Rango temporal       : {df_final.index[0]} → {df_final.index[-1]}")
print(f"  Fechas duplicadas    : {df_final.index.duplicated().sum()}"
      if df_final.index.duplicated().sum() == 0
      else f"  Fechas duplicadas: {df_final.index.duplicated().sum()}")

# Mostrar NaN tras la unión
nulos_union = df_final.isnull().sum()
print(f"\n  NaN por variable tras la unión:")
for col, n in nulos_union[nulos_union > 0].items():
    pct = round(n / len(df_final) * 100, 1)
    print(f"    · {col:<35} {n:>4} ({pct:>5}%)")
print(f"\n  Total NaN tras unión : {nulos_union.sum()}")
print(f"  Variables con NaN    : {(nulos_union > 0).sum()}")

# df.describe() del dataset integrado
print(f"\n  df.describe() (dataset integrado):")
print(df_final.describe().round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4.3 — DETECCIÓN Y TRATAMIENTO DE NA
#
# Los NaN en el dataset integrado tienen dos orígenes:
#
# 1. NaN estructurales por lags y YoY 
# 2. NaN por desajuste de cobertura temporal entre fuentes 
#
#   · Primero interpolamos los NaN de desajuste temporal
#   · Después eliminamos con dropna() los NaN estructurales de lags/YoY
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 4.3 — DETECCIÓN Y TRATAMIENTO DE NA")
print("="*65)

# Identificar columnas con NaN que NO son de lags/YoY
cols_lags_yoy = [c for c in df_final.columns
                 if 'lag' in c or 'yoy' in c]
cols_base     = [c for c in df_final.columns
                 if c not in cols_lags_yoy]

nulos_base = df_final[cols_base].isnull().sum()
if nulos_base.sum() > 0:
    print(f"\n  NaN en variables base (posible desajuste temporal):")
    for col, n in nulos_base[nulos_base > 0].items():
        print(f"    · {col}: {n}")
    print("  → Aplicando interpolación temporal...")
    for col in cols_base:
        if df_final[col].isnull().sum() > 0:
            df_final[col] = df_final[col].interpolate(
                method='time', limit_direction='both')
    print(f"  NaN en variables base tras interpolación: "
          f"{df_final[cols_base].isnull().sum().sum()}")
else:
    print(f"\n  Variables base: 0 NaN (coberturas temporales alineadas)")

# NaN restantes son estructurales (lags y YoY) — se documentan
nulos_restantes = df_final.isnull().sum()
print(f"\n  NaN estructurales (lags + YoY) — se eliminan con dropna():")
for col, n in nulos_restantes[nulos_restantes > 0].items():
    print(f"    · {col:<35} {n:>4} NaN")
print(f"\n  Total NaN estructurales: {nulos_restantes.sum()}")
print(f"  Origen: pct_change(4) → 4 trimestres, shift(n) → n trimestres")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4.4 — ELIMINACIÓN DE REGISTROS INCOMPLETOS (dropna)
#
# Eliminar los primeros trimestres que tienen
# NaN estructurales por los lags y el pct_change(4).
# El lag máximo es lag4 → se pierden 4 trimestres adicionales al YoY
# (que ya pierde 4). En total se pierden los primeros ~8 trimestres.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 4.4 — ELIMINACIÓN DE REGISTROS INCOMPLETOS (dropna)")
print("="*65)

n_antes = len(df_final)
df_final = df_final.dropna()
n_eliminados = n_antes - len(df_final)

print(f"\n  Observaciones antes de dropna : {n_antes}")
print(f"  Observaciones eliminadas      : {n_eliminados}")
print(f"  Observaciones tras dropna     : {len(df_final)}")
print(f"  Primer trimestre válido       : {df_final.index[0]}")
print(f"  Último trimestre válido       : {df_final.index[-1]}")
print(f"\n  Motivo: los primeros trimestres tienen NaN estructurales")
print(f"  por el pct_change(4) (YoY) y el shift(4) (lag4).")
print(f"  Se conservan {len(df_final)} trimestres completos para el modelo.")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4.5 — VALIDACIÓN FINAL DEL DATASET
#
# Comprobación completa:
# df.head(), df.tail(), df.describe(), df.isna().sum(), df.shape,
# df.dtypes, df.index, df.duplicated() y resumen_dataset() propia.
# Validación adicional con ndarray numpy.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 4.5 — VALIDACIÓN FINAL DEL DATASET")
print("="*65)

# df.shape
print(f"\n  df.shape    : {df_final.shape}")

# df.dtypes
print(f"\n  df.dtypes:")
print(df_final.dtypes.to_string())

# df.index
print(f"\n  df.index:")
print(f"    Tipo    : {type(df_final.index)}")
print(f"    Rango   : {df_final.index[0]} → {df_final.index[-1]}")
print(f"    Freq    : trimestral")

# df.duplicated()
n_dup = df_final.index.duplicated().sum()
print(f"\n  df.duplicated(): {n_dup}" if n_dup == 0
      else f"\n  Duplicados: {n_dup}")

# df.isna().sum()
print(f"\n  df.isna().sum(): {df_final.isna().sum().sum()}"
      if df_final.isna().sum().sum() == 0
      else f"\n  NaN: {df_final.isna().sum().sum()}")

# df.head() y df.tail()
print(f"\n  df.head(3):")
print(df_final.head(3).to_string())
print(f"\n  df.tail(3):")
print(df_final.tail(3).to_string())

# df.describe()
print(f"\n  df.describe():")
print(df_final.describe().round(2).to_string())

# Validación con ndarray
arr = df_final.values
print(f"\n  Validación con numpy (ndarray):")
print(f"    · shape             : {arr.shape}")
print(f"    · np.isnan() total  : {np.isnan(arr).sum()}"
      if np.isnan(arr).sum() == 0
      else f"    · NaN: {np.isnan(arr).sum()}")
print(f"    · Registros incom.  : {np.isnan(arr).any(axis=1).sum()}"
      if np.isnan(arr).any(axis=1).sum() == 0
      else f"    · Registros incompletos: {np.isnan(arr).any(axis=1).sum()}")

# Resumen completo con función propia
resumen_dataset("Dataset macro-financiero final VALIDADO", df_final)

# ── Gráfico G1: completitud del dataset final ─────────────────────────────────
# Confirma visualmente que el 100% de las variables tienen completitud total
completitud = (1 - df_final.isna().mean()) * 100

fig, ax = plt.subplots(figsize=(12, 10))
colores_barras = []
for col in completitud.index:
    if col in df_bde.columns:
        colores_barras.append("#2196F3")   # azul BdE
    elif col in df_ine.columns:
        colores_barras.append("#4CAF50")   # verde INE
    else:
        colores_barras.append("#FF9800")   # naranja petróleo

bars = ax.barh(completitud.index, completitud.values,
               color=colores_barras, alpha=0.8, edgecolor="white")
ax.set_xlim(0, 115)
ax.set_xlabel("% de datos válidos (completitud)", fontsize=11)
ax.set_title("Dataset macro-financiero final — Completitud por variable\n"
             "(azul=BdE, verde=INE, naranja=Petróleo)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, completitud.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")
ax.axvline(100, color="green", linestyle="--", linewidth=1.2,
           alpha=0.7, label="100% completitud")
ax.legend(fontsize=9)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G1_completitud_dataset_final.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G1 guardado: {ruta}")

# ── Gráfico G2: series temporales variables objetivo y principales ─────────────
# Vista general de las variables más importantes del dataset final
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Dataset macro-financiero final — Variables principales\n"
             "(dataset validado, trimestral)",
             fontsize=13, fontweight="bold")

vars_plot = [
    ("mora_hogares",    "Mora Hogares (miles €)",          "#2196F3"),
    ("tasa_paro",       "Tasa de Paro (%)",                "#E91E63"),
    ("euribor_12m",     "Euribor 12m (%)",                 "#9C27B0"),
    ("pib_yoy",         "PIB YoY (%)",                     "#4CAF50"),
    ("brent_yoy",       "Brent YoY (%)",                   "#FF9800"),
    ("ratio_credito_pib","Ratio Crédito/PIB (%)",          "#00BCD4"),
]

for ax, (col, etiqueta, color) in zip(axes.flat, vars_plot):
    if col in df_final.columns:
        ax.plot(df_final.index, df_final[col],
                color=color, linewidth=1.8, marker="o", markersize=2)
        if df_final[col].min() < 0:
            ax.axhline(0, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5)
        ax.set_title(etiqueta, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G2_series_temporales_final.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G2 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR DATASET FINAL
# ══════════════════════════════════════════════════════════════════════════════

ruta_output = os.path.join(TRANSF_PATH, "dataset_final.xlsx")
df_export = df_final.copy()
df_export.index = df_export.index.strftime('%Y-%m-%d')
df_export.index.name = 'fecha'
df_export.to_excel(ruta_output)
print(f"\n  dataset_final.xlsx guardado: {ruta_output}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — DATASET MACRO-FINANCIERO VALIDADO")
print("="*65)
print(f"\n  {'Bloque':<12} {'Variables':<5} {'Fuente'}")
print(f"  {'-'*12} {'-'*5} {'-'*30}")
print(f"  {'BdE':<12} {len([c for c in df_final.columns if c in df_bde.columns]):<5} "
      f"Banco de España")
print(f"  {'INE':<12} {len([c for c in df_final.columns if c in df_ine.columns]):<5} "
      f"Instituto Nacional de Estadística")
print(f"  {'Petróleo':<12} {len([c for c in df_final.columns if c in df_pet.columns]):<5} "
      f"World Bank Pink Sheet")

print(f"\n  Shape final          : {df_final.shape}")
print(f"  Nº variables         : {df_final.shape[1]}")
print(f"  Nº observaciones     : {df_final.shape[0]} trimestres")
print(f"  Rango temporal       : {df_final.index[0]} → {df_final.index[-1]}")
print(f"  Frecuencia           : trimestral")
print(f"  Total NaN            : {df_final.isna().sum().sum()}")
print(f"  Variable objetivo    : mora_hogares")
print(f"\n  Dataset macro-financiero final VALIDADO.")
print(f"     Output  : {ruta_output}")
print(f"     Gráficos:")
print(f"       G1  Completitud por variable (100% todas)")
print(f"       G2  Series temporales variables principales")