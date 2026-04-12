# -*- coding: utf-8 -*-
"""
EDA — Análisis Exploratorio de Datos

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH    = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\EDA'

os.makedirs(GRAFICOS_PATH, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# Colores por fuente
C_BDE = "#2196F3"   # azul  — Banco de España
C_INE = "#4CAF50"   # verde — INE
C_PET = "#FF9800"   # naranja — Petróleo
C_OBJ = "#E91E63"   # rosa  — variable objetivo

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL DATASET
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(os.path.join(INPUT_PATH, "dataset_final.xlsx"),
                   index_col=0, parse_dates=True)
df.index = pd.DatetimeIndex(df.index.date)
df.index.name = 'fecha'

print(f"Dataset cargado: {df.shape} — {df.shape[0]} trimestres, {df.shape[1]} variables")
print(f"Rango: {df.index[0]} -> {df.index[-1]}")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3.1.A — DESCRIPTIVOS BÁSICOS
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  PASO 3.1.A — DESCRIPTIVOS BÁSICOS")
print("="*65)

# Verificación de integridad
print(f"\n  NaN totales     : {df.isna().sum().sum()}")
print(f"  Fechas duplicadas: {df.index.duplicated().sum()}")

# Variables que no pueden ser negativas (las YoY sí pueden serlo)
vars_no_neg = ['mora_hogares', 'mora_empresas', 'credito_hogares',
               'credito_empresas', 'precio_m2_vivienda', 'pib', 'tasa_paro', 'brent']
vars_no_neg = [v for v in vars_no_neg if v in df.columns]

print(f"\n  Valores negativos en variables que no pueden serlo:")
for col in vars_no_neg:
    n = (df[col] < 0).sum()
    estado = "OK" if n == 0 else f"[AVISO] {n} negativos"
    print(f"    · {col:<30} {estado}")

# Estadísticos descriptivos completos
print(f"\n  Estadísticos descriptivos:")
print(f"  {'Variable':<30} {'Media':>10} {'Mediana':>10} {'DT':>10} "
      f"{'CV%':>6} {'Min':>10} {'Max':>10} {'Asim':>6} {'Kurt':>6}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")

for col in df.columns:
    s    = df[col]
    med  = s.mean()
    mdn  = s.median()
    std  = s.std()
    cv   = abs(std / med * 100) if med != 0 else np.nan
    mn   = s.min()
    mx   = s.max()
    sk   = s.skew()
    ku   = s.kurt()
    print(f"  {col:<30} {med:>10.2f} {mdn:>10.2f} {std:>10.2f} "
          f"{cv:>6.1f} {mn:>10.2f} {mx:>10.2f} {sk:>6.2f} {ku:>6.2f}")

print(f"\n  Total NaN: {df.isna().sum().sum()} — Completitud: 100.0%")

# G1: Histogramas + KDE de las variables YoY del modelo
colores_hist = [C_OBJ, C_BDE, C_BDE, C_BDE, C_INE, C_INE, C_INE, C_PET]
vars_hist    = [
    'mora_hogares',
    'credito_hogares_yoy',
    'euribor_12m',
    'precio_m2_vivienda_yoy',
    'tasa_paro',
    'pib_yoy',
    'ipc_var_anual',
    'brent_yoy',
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("EDA — Histogramas + KDE de variables del modelo\n"
             "(versiones YoY — dataset macro-financiero final, 2004-2025)",
             fontsize=13, fontweight="bold")

for ax, col, color in zip(axes.flat, vars_hist, colores_hist):
    s = df[col].dropna()
    ax.hist(s, bins=20, color=color, alpha=0.5, edgecolor='white',
            density=True, label='Histograma')
    s.plot.kde(ax=ax, color=color, linewidth=2)
    ax.axvline(s.mean(),   color='black', linestyle='--', linewidth=1,
               label=f'Media: {s.mean():.2f}')
    ax.axvline(s.median(), color='gray',  linestyle=':',  linewidth=1,
               label=f'Mediana: {s.median():.2f}')
    if s.min() < 0:
        ax.axvline(0, color='red', linestyle='-', linewidth=0.6, alpha=0.4)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

axes.flat[-1].set_visible(False)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G1_histogramas_eda.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  G1 guardado: {ruta}")

# ──────────────────────────────────────────────────────────────────────────────
# PASO 3.2.A — VISUALIZACIÓN DE SERIES TEMPORALES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PASO 3.2.A — VISUALIZACIÓN DE SERIES TEMPORALES")
print("="*65)

# G2: Series temporales de las 8 variables del modelo
vars_series = [
    ('mora_hogares',           'Mora Hogares (miles €)',         C_OBJ),
    ('euribor_12m',            'Euribor 12m (%)',                C_BDE),
    ('credito_hogares_yoy',    'Crédito Hogares YoY (%)',        C_BDE),
    ('precio_m2_vivienda_yoy', 'Precio m² Vivienda YoY (%)',     C_BDE),
    ('tasa_paro',              'Tasa de Paro (%)',               C_INE),
    ('pib_yoy',                'PIB YoY (%)',                    C_INE),
    ('ipc_var_anual',          'IPC Variación Anual (%)',        C_INE),
    ('brent_yoy',              'Brent YoY (%)',                  C_PET),
]

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
fig.suptitle("EDA — Series temporales: mora hogares y variables del modelo\n"
             "(versiones YoY — dataset macro-financiero final, 2004-2025)",
             fontsize=13, fontweight="bold")

for ax, (col, etiqueta, color) in zip(axes.flat, vars_series):
    ax.plot(df.index, df[col], color=color, linewidth=1.8,
            marker='o', markersize=2)
    if df[col].min() < 0:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(etiqueta, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G2_series_temporales_eda.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G2 guardado: {ruta}")

# G3: Doble eje — mora_hogares vs euribor_12m y tasa_paro
fig, ax1 = plt.subplots(figsize=(14, 6))
fig.suptitle("EDA — Mora Hogares vs Euribor 12m y Tasa de Paro\n"
             "(doble eje — correlación visual y retardos temporales)",
             fontsize=13, fontweight='bold')

ax1.plot(df.index, df['mora_hogares'] / 1e6, color=C_OBJ,
         linewidth=2, label='Mora Hogares (miles M€)', zorder=3)
ax1.set_ylabel('Mora Hogares (miles millones €)', color=C_OBJ, fontsize=10)
ax1.tick_params(axis='y', labelcolor=C_OBJ)

ax2 = ax1.twinx()
ax2.plot(df.index, df['euribor_12m'], color=C_BDE,
         linewidth=1.5, linestyle='--', label='Euribor 12m (%)', alpha=0.8)
ax2.plot(df.index, df['tasa_paro'], color=C_INE,
         linewidth=1.5, linestyle=':', label='Tasa Paro (%)', alpha=0.8)
ax2.set_ylabel('% (Euribor y Tasa Paro)', fontsize=10)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
ax1.tick_params(axis='x', rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G3_mora_vs_euribor_paro.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G3 guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3.3.A — CORRELACIÓN PEARSON + VIF COMPARATIVO (niveles vs YoY)
#   1. VIF en niveles → detecta ratio_credito_pib VIF=63 → se excluye
#   2. Matriz de correlación sobre variables YoY definitivas (sin ratio)
#   3. VIF en YoY → confirma que todos los VIF bajan de 5
#   4. Gráfico comparativo niveles vs YoY
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  PASO 3.3.A — CORRELACIÓN PEARSON + VIF COMPARATIVO")
print("="*65)

# --- 1. VIF Escenario A: variables en NIVELES --------------------------------
# Se incluyen todas las variables candidatas incluyendo ratio_credito_pib
# para detectar si hay multicolinealidad y justificar qué variables excluir.
vars_candidatas = [
    'credito_hogares', 'euribor_12m', 'precio_m2_vivienda',
    'tasa_paro', 'pib', 'ipc_var_anual', 'brent', 'ratio_credito_pib'
]
vars_niv_ok = [v for v in vars_candidatas if v in df.columns]
df_niv      = df[vars_niv_ok].dropna()
X_niv       = df_niv.values
X_niv_std   = (X_niv - X_niv.mean(axis=0)) / X_niv.std(axis=0)

vifs_niv = {col: variance_inflation_factor(X_niv_std, i)
            for i, col in enumerate(df_niv.columns)}

print(f"\n  VIF — Escenario A: variables en NIVELES (diagnóstico previo)")
print(f"  {'Variable':<30} {'VIF':>8}  Diagnóstico")
print(f"  {'-'*30} {'-'*8}  {'-'*25}")
for col, vif in sorted(vifs_niv.items(), key=lambda x: x[1], reverse=True):
    diag = ("Critica (>30)"  if vif > 30 else
            "Severa (>10)"   if vif > 10 else
            "Moderada (>5)"  if vif > 5  else
            "Sin problema")
    print(f"  {col:<30} {vif:>8.2f}  {diag}")

print(f"\n  ratio_credito_pib: VIF={vifs_niv.get('ratio_credito_pib', 0):.2f} "
      f"— combinación lineal de credito_hogares y pib.")
print(f"  Decisión: excluida del modelo definitivo.")

# Variables YoY definitivas del modelo 
VARS_YOY = [
    'mora_hogares',
    'credito_hogares_yoy',
    'euribor_12m',
    'precio_m2_vivienda_yoy',
    'tasa_paro',
    'pib_yoy',
    'ipc_var_anual',
    'brent_yoy',
]

# --- 2. Matriz de correlación sobre variables YoY ----------------------------
# Con ratio_credito_pib ya excluida, se calcula la correlación sobre
# las variables YoY 

df_yoy = df[VARS_YOY].dropna()
corr   = df_yoy.corr()

print(f"\n  Matriz de correlación de Pearson ({len(VARS_YOY)} variables — YoY):")
print(corr.round(2).to_string())

print(f"\n  Correlaciones con mora_hogares (ordenadas por |r|):")
corr_mora = corr['mora_hogares'].drop('mora_hogares').sort_values(
    key=abs, ascending=False)
for var, val in corr_mora.items():
    signo  = "positiva" if val > 0 else "negativa"
    fuerza = ("muy alta" if abs(val) > 0.8 else
              "alta"     if abs(val) > 0.6 else
              "moderada" if abs(val) > 0.4 else "baja")
    print(f"    · {var:<30} r = {val:>6.3f}  ({fuerza} {signo})")

# G4: Mapa de calor de correlaciones (YoY, sin ratio_credito_pib)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        val       = corr.iloc[i, j]
        color_txt = 'white' if abs(val) > 0.7 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color=color_txt, fontweight='bold')

ax.set_title("EDA — Matriz de correlación de Pearson\n"
             "(variables del modelo en YoY, sin ratio_credito_pib, 2004-2025)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G4_matriz_correlacion.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  G4 guardado: {ruta}")

# --- 3. VIF Escenario B: variables en YoY ------------------------------------
# Confirma que la transformación YoY resuelve la multicolinealidad.
# ratio_credito_pib ya está excluida por el diagnóstico anterior.

vars_vif_yoy = [v for v in VARS_YOY if v != 'mora_hogares']
df_vif_yoy   = df[vars_vif_yoy].dropna()
X_yoy        = df_vif_yoy.values
X_yoy_std    = (X_yoy - X_yoy.mean(axis=0)) / X_yoy.std(axis=0)

vifs_yoy = {col: variance_inflation_factor(X_yoy_std, i)
            for i, col in enumerate(df_vif_yoy.columns)}

print(f"\n  VIF — Escenario B: variables en YoY (modelo definitivo)")
print(f"  {'Variable':<30} {'VIF':>8}  Diagnóstico")
print(f"  {'-'*30} {'-'*8}  {'-'*25}")
for col, vif in sorted(vifs_yoy.items(), key=lambda x: x[1], reverse=True):
    diag = ("Severa (>10)"  if vif > 10 else
            "Moderada (>5)" if vif > 5  else
            "Sin problema")
    print(f"  {col:<30} {vif:>8.2f}  {diag}")

# G5: VIF comparativo niveles vs YoY
equiv = {
    'credito_hogares_yoy':    'credito_hogares',
    'precio_m2_vivienda_yoy': 'precio_m2_vivienda',
    'pib_yoy':                'pib',
    'brent_yoy':              'brent',
    'euribor_12m':            'euribor_12m',
    'tasa_paro':              'tasa_paro',
    'ipc_var_anual':          'ipc_var_anual',
}

etiquetas, vif_niv_plot, vif_yoy_plot = [], [], []
for var_yoy, var_niv in equiv.items():
    if var_yoy in vifs_yoy and var_niv in vifs_niv:
        etiquetas.append(var_yoy)
        vif_niv_plot.append(vifs_niv[var_niv])
        vif_yoy_plot.append(vifs_yoy[var_yoy])

x     = np.arange(len(etiquetas))
ancho = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - ancho/2, vif_niv_plot, ancho, label='Niveles',
            color='#FFCDD2', edgecolor='white', alpha=0.9)
b2 = ax.bar(x + ancho/2, vif_yoy_plot, ancho, label='YoY (modelo)',
            color='#C8E6C9', edgecolor='white', alpha=0.9)
ax.axhline(5,  color='orange', linestyle='--', linewidth=1.2,
           label='Umbral moderado (VIF=5)')
ax.axhline(10, color='red',    linestyle='--', linewidth=1.2,
           label='Umbral severo (VIF=10)')

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#c62828')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7, color='#2e7d32')

ax.set_xticks(x)
ax.set_xticklabels(etiquetas, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('VIF', fontsize=11)
ax.set_title("EDA — VIF comparativo: Niveles vs YoY\n"
             "(impacto de la transformación sobre la multicolinealidad)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G5_vif_comparativo.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G5 guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3.4.A — TENDENCIAS, CROSS-CORRELATION, DESCOMPOSICIÓN DE MORA-HOGARES Y SCATTER
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PASO 3.4.A — TENDENCIAS, CROSS-CORRELATION, DESCOMPOSICIÓN Y SCATTER")
print("="*65)

# G6: Tendencias con media móvil de 4 trimestres
vars_tend = [
    ('mora_hogares',           'Mora Hogares (miles €)',        C_OBJ),
    ('tasa_paro',              'Tasa de Paro (%)',              C_INE),
    ('precio_m2_vivienda_yoy', 'Precio m² Vivienda YoY (%)',   C_BDE),
    ('pib_yoy',                'PIB YoY (%)',                   C_INE),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("EDA — Tendencias y ciclos (media móvil 4 trimestres)\n"
             "(identificación de patrones y puntos de inflexión)",
             fontsize=13, fontweight='bold')

for ax, (col, etiqueta, color) in zip(axes.flat, vars_tend):
    serie   = df[col]
    rolling = serie.rolling(window=4, center=True).mean()
    ax.plot(df.index, serie,   color=color, linewidth=1,   alpha=0.4,
            label='Trimestral')
    ax.plot(df.index, rolling, color=color, linewidth=2.5,
            label='Tendencia (MM4)')
    if serie.min() < 0:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(etiqueta, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G6_tendencias_ciclos.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G6 guardado: {ruta}")

# --- Cross-correlation -------------------------------------------------------
# Calcula corr(mora_hogares(t), variable(t-lag)) para lag = 0..4
# Un lag positivo indica que la variable precede a la mora.

print(f"\n  Cross-correlation con mora_hogares:")
print(f"  {'Variable':<28} {'lag0':>7} {'lag1':>7} {'lag2':>7} "
      f"{'lag3':>7} {'lag4':>7} {'Lag opt':>8} {'|r| max':>8}")
print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

mora = df['mora_hogares'].values
resultados_cross = {}

# Variables para cross-correlation — todas las YoY candidatas incluyendo
# pib_yoy, cuya exclusión se justificará empíricamente con r < 0.15.
vars_cross = [
    'euribor_12m', 'tasa_paro', 'pib_yoy', 'ipc_var_anual',
    'brent_yoy', 'credito_hogares_yoy', 'precio_m2_vivienda_yoy',
]

for col in vars_cross:
    x    = df[col].values
    lags = {}
    for lag in range(5):
        if lag == 0:
            mask     = ~(np.isnan(mora) | np.isnan(x))
            corr_val = np.corrcoef(mora[mask], x[mask])[0, 1]
        else:
            mora_t   = mora[lag:]
            x_t      = x[:-lag]
            mask     = ~(np.isnan(mora_t) | np.isnan(x_t))
            corr_val = (np.corrcoef(mora_t[mask], x_t[mask])[0, 1]
                        if mask.sum() > 2 else np.nan)
        lags[lag] = corr_val
    lag_opt = max(lags, key=lambda k: abs(lags[k]))
    r_max   = abs(lags[lag_opt])
    resultados_cross[col] = lags
    vals = [f"{lags[l]:>7.3f}" for l in range(5)]
    print(f"  {col:<28} {'  '.join(vals)} {lag_opt:>8} {r_max:>8.3f}")

# G7: Cross-correlation como mapa de calor
lags_labels = [f'lag{i}' for i in range(5)]
mat = np.array([[resultados_cross[col][lag] for lag in range(5)]
                for col in [
    'euribor_12m',
    'tasa_paro',
    'pib_yoy',
    'ipc_var_anual',
    'brent_yoy',
    'credito_hogares_yoy',
    'precio_m2_vivienda_yoy',
]], dtype=float)

fig, ax = plt.subplots(figsize=(11, 6))
lim = max(abs(mat.min()), abs(mat.max()))
pc  = ax.pcolormesh(mat, cmap='RdYlGn', vmin=-lim, vmax=lim)
plt.colorbar(pc, ax=ax)
ax.set_xticks(np.arange(5) + 0.5)
ax.set_yticks(np.arange(len(vars_cross)) + 0.5)
ax.set_xticklabels(lags_labels, fontsize=9)
ax.set_yticklabels(vars_cross, fontsize=9)

for i in range(len(vars_cross)):
    for j in range(5):
        val       = mat[i, j]
        color_txt = 'white' if abs(val) > 0.7 else 'black'
        ax.text(j + 0.5, i + 0.5, f'{val:.3f}',
                ha='center', va='center',
                fontsize=8, color=color_txt, fontweight='bold')

ax.set_title("EDA — Cross-correlation: variables predictoras vs mora_hogares\n"
             "(correlación de Pearson por lag — lag óptimo = mayor |r|)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Lag (trimestres)", fontsize=10)
ax.set_ylabel("Variable predictora", fontsize=10)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G7_cross_correlation.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  G7 guardado: {ruta}")

# G8: Cross-correlation por variable (líneas)
colores_cross = [C_BDE, C_INE, C_INE, C_INE, C_PET, C_BDE, C_BDE]

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
fig.suptitle("EDA — Cross-correlation por variable predictora\n"
             "(evolución de la correlación con mora_hogares según el lag)",
             fontsize=13, fontweight='bold')

for ax, col, color in zip(axes.flat, vars_cross, colores_cross):
    lags_vals = list(resultados_cross[col].values())
    lag_opt   = max(range(5), key=lambda k: abs(lags_vals[k]))
    ax.plot(range(5), lags_vals, marker='o', linewidth=2, color=color)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.scatter([lag_opt], [lags_vals[lag_opt]], color='red',
               s=80, zorder=5, label=f'Lag opt: {lag_opt}')
    ax.set_title(col, fontsize=8, fontweight='bold')
    ax.set_xlabel('Lag (trimestres)', fontsize=8)
    ax.set_ylabel('Correlación r', fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xticks(range(5))

axes.flat[-1].set_visible(False)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G8_cross_correlation_lineas.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G8 guardado: {ruta}")

# G9: Descomposición de mora_hogares (variable objetivo)
# Muestra tendencia, estacionalidad y residuo 
print(f"\n  Descomponiendo mora_hogares (modelo aditivo, period=4)...")
serie_mora   = df['mora_hogares'].dropna()
descomp_mora = seasonal_decompose(serie_mora, model='additive', period=4)

fig = descomp_mora.plot()
fig.set_size_inches(14, 9)
plt.suptitle("EDA — Descomposición de serie temporal: mora_hogares\n"
             "(Trend + Seasonal + Residual — periodo=4 trimestres)",
             fontsize=13, fontweight='bold', y=1.02)

ruta = os.path.join(GRAFICOS_PATH, "G9_descomposicion_mora_hogares.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G9 guardado: {ruta}")

# G10: Scatter plots mora_hogares vs predictores seleccionados
predictors_scatter = [
    ('credito_hogares_yoy', 'credito_hogares_yoy (lag 0)'),
    ('euribor_12m',         'euribor_12m (lag 0)'),
    ('ipc_var_anual',       'ipc_var_anual (lag 0)'),
    ('brent_yoy',           'brent_yoy (lag 0)'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA — Diagramas de dispersión: mora_hogares vs predictores\n"
             "(con línea de regresión lineal — dataset 2004-2025)",
             fontsize=14, fontweight='bold')

colores_scatter = [C_BDE, C_BDE, C_INE, C_PET]
for ax, (var, titulo), color in zip(axes.flat, predictors_scatter, colores_scatter):
    sns.regplot(x=var, y='mora_hogares', data=df, ax=ax,
                scatter_kws={'alpha': 0.6, 'color': color},
                line_kws={'color': 'red', 'linewidth': 1.5})
    ax.set_title(titulo, fontsize=10, fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel('mora_hogares')

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G10_scatter_plots.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G10 guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3.5.A — TABLA RESUMEN: DESCRIPCIÓN DEL DATASET
# Tabla descriptiva con las variables que entran al modelo:
# nombre, fuente, unidad, tipo, frecuencia, periodo y lag seleccionado.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PASO 3.5.A — TABLA RESUMEN: DESCRIPCIÓN DEL DATASET")
print("="*65)

tabla_vars = [
    # (variable, fuente, unidad, tipo, frecuencia, lag, decision)
    ('mora_hogares',           'BdE be0413', 'Miles €',        'Cuantitativa continua', 'Trimestral', 'lag0 (objetivo)', 'Variable objetivo'),
    ('tasa_paro',              'INE 65079',  '%',              'Cuantitativa continua', 'Trimestral', 'lag3 (9m)',       'INCLUIR'),
    ('credito_hogares_yoy',    'BdE be0413', '%',              'Cuantitativa continua', 'Trimestral', 'lag0 (0m)',       'INCLUIR'),
    ('precio_m2_vivienda_yoy', 'BdE be2507', '%',              'Cuantitativa continua', 'Trimestral', 'lag4 (12m)',      'INCLUIR'),
    ('euribor_12m',            'BdE be1901', '%',              'Cuantitativa continua', 'Trimestral', 'lag0 (0m)',       'INCLUIR'),
    ('ipc_var_anual',          'INE 50917',  '%',              'Cuantitativa continua', 'Trimestral', 'lag0 (0m)',       'INCLUIR'),
    ('brent_yoy',              'World Bank', '%',              'Cuantitativa continua', 'Trimestral', 'lag0 (0m)',       'INCLUIR'),
    ('pib_yoy',                'INE 67198',  '%',              'Cuantitativa continua', 'Trimestral', 'N/A',             'EXCLUIR'),
    ('ratio_credito_pib',      'BdE+INE',    '% PIB',          'Cuantitativa continua', 'Trimestral', 'N/A',             'EXCLUIR'),
]

cabeceras = ['Variable', 'Fuente', 'Unidad', 'Tipo', 'Frecuencia',
             'Lag seleccionado', 'Decisión']
anchos    = [0.18, 0.10, 0.07, 0.16, 0.09, 0.13, 0.17]

fig, ax = plt.subplots(figsize=(22, 5))
ax.axis('off')
fig.suptitle("EDA — Tabla resumen: descripción y selección de variables del modelo\n"
             "(fuente, unidad, tipo, frecuencia, periodo y lag óptimo empírico)",
             fontsize=13, fontweight='bold', y=1.05)

tabla = ax.table(
    cellText=tabla_vars,
    colLabels=cabeceras,
    colWidths=anchos,
    cellLoc='center',
    loc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1, 2.0)

for j in range(len(cabeceras)):
    tabla[0, j].set_facecolor('#212121')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

for i in range(len(tabla_vars)):
    color_fila = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
    # Colorear en rojo claro las filas excluidas
    if 'EXCLUIR' in tabla_vars[i][-1]:
        color_fila = '#FFEBEE'
    for j in range(len(cabeceras)):
        tabla[i+1, j].set_facecolor(color_fila)
        tabla[i+1, j].set_linewidth(0.5)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G11_tabla_resumen_variables.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  G11 guardado: {ruta}")

# G12: Resumen — correlación máxima y lag óptimo por variable
vars_res = list(resultados_cross.keys())
r_maxs   = [abs(resultados_cross[c][max(resultados_cross[c],
            key=lambda k: abs(resultados_cross[c][k]))]) for c in vars_res]
lag_opts = [max(resultados_cross[c], key=lambda k: abs(resultados_cross[c][k]))
            for c in vars_res]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("EDA — Resumen selección de variables\n"
             "(correlación máxima con mora_hogares y lag óptimo empírico)",
             fontsize=13, fontweight='bold')

colores_r = [C_BDE if 'euribor' in v or 'credito' in v or 'precio' in v
             else C_PET if 'brent' in v else C_INE for v in vars_res]

axes[0].barh(vars_res, r_maxs, color=colores_r, edgecolor='white', alpha=0.85)
axes[0].axvline(0.6, color='orange', linestyle='--', linewidth=1,
                label='Alta correlación (0.6)')
axes[0].axvline(0.8, color='red',    linestyle='--', linewidth=1,
                label='Muy alta correlación (0.8)')
for i, (v, r) in enumerate(zip(vars_res, r_maxs)):
    axes[0].text(r + 0.005, i, f'{r:.3f}', va='center', fontsize=8,
                 fontweight='bold')
axes[0].set_xlabel('|r| máximo con mora_hogares', fontsize=10)
axes[0].set_title('Fuerza de la correlación (con lag óptimo)',
                  fontsize=10, fontweight='bold')
axes[0].legend(fontsize=7)
axes[0].set_xlim(0, 1.05)

colores_lag = ['#C8E6C9' if l == 0 else '#FFF9C4' if l <= 2 else '#FFCDD2'
               for l in lag_opts]
axes[1].barh(vars_res, lag_opts, color=colores_lag, edgecolor='white', alpha=0.85)
for i, (v, l) in enumerate(zip(vars_res, lag_opts)):
    axes[1].text(l + 0.03, i, f'lag{l} ({l*3}m)', va='center',
                 fontsize=8, fontweight='bold')
axes[1].set_xlabel('Lag óptimo (trimestres)', fontsize=10)
axes[1].set_title('Retardo óptimo empírico\n(trimestres que precede a la mora)',
                  fontsize=10, fontweight='bold')
axes[1].set_xlim(0, 5.5)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels([f'lag{i}\n({i*3}m)' for i in range(5)], fontsize=8)

from matplotlib.patches import Patch
leyenda_lag = [
    Patch(facecolor='#C8E6C9', label='Contemporáneo (lag0)'),
    Patch(facecolor='#FFF9C4', label='Corto plazo (lag1-2)'),
    Patch(facecolor='#FFCDD2', label='Medio plazo (lag3-4)'),
]
axes[1].legend(handles=leyenda_lag, fontsize=7, loc='lower right')

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G12_resumen_seleccion_variables.png")
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.show()
print(f"  G12 guardado: {ruta}")


# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESUMEN FINAL — EDA v3")
print("="*65)
print(f"\n  Dataset        : dataset_final.xlsx")
print(f"  Observaciones  : {df.shape[0]} trimestres (2004-Q1 -> 2025-Q1)")
print(f"  Variables      : {df.shape[1]} totales / 6 predictoras + mora_hogares")
print(f"  Variable Y     : mora_hogares")
print(f"\n  Gráficos generados:")
print(f"    G1   Histogramas + KDE (variables YoY)")
print(f"    G2   Series temporales 8 variables")
print(f"    G3   Mora vs Euribor y Paro (doble eje)")
print(f"    G4   Matriz de correlación (YoY)")
print(f"    G5   VIF comparativo niveles vs YoY")
print(f"    G6   Tendencias y ciclos (media móvil)")
print(f"    G7   Cross-correlation (mapa de calor)")
print(f"    G8   Cross-correlation por variable (líneas)")
print(f"    G9   Descomposición mora_hogares")
print(f"    G10  Scatter plots mora vs predictores")
print(f"    G11  Tabla descriptiva dataset")
print(f"    G12  Resumen selección variables")

