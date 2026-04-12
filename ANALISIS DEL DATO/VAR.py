# -*- coding: utf-8 -*-
"""
ANÁLISIS DEL DATO —  VAR(1)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR, adfuller
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

RUTA_DATOS_ANALISIS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\analisis_del_dato'
RUTA_GRAFICOS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\VAR'

os.makedirs(RUTA_GRAFICOS, exist_ok=True)

# Estilo visual
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams['font.size'] = 10

COLOR_PRINCIPAL = '#003366'   # Azul oscuro BdE
COLOR_TEST      = '#d62728'   # Rojo
COLOR_NEUTRO    = '#1f77b4'   # Azul medio

print("\n" + "="*80)
print("PASO 3: VAR(1) — ANÁLISIS COMPLEMENTARIO")
print("="*80)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

# Cargamos el dataset ya escalado (mora_hogares en M€) generado por OLS.py
ruta_datos = os.path.join(RUTA_DATOS_ANALISIS, 'dataset_modelos.xlsx')
df = pd.read_excel(ruta_datos)

print(f"\n  Dataset cargado: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(f"  mora_hogares — Min: {df['mora_hogares'].min():.1f} M€  "
      f"Max: {df['mora_hogares'].max():.1f} M€  "
      f"Media: {df['mora_hogares'].mean():.1f} M€")
print(f"  (verificación escala: valores esperados entre 2.777 M€ y 50.874 M€)")

# El VAR incluye todas las variables del sistema, tratándolas como endógenas.
# mora_hogares se incluye al final para que sea la variable de respuesta
# en las funciones impulso-respuesta.
VARIABLES_VAR = [
    'tasa_paro_lag3',
    'credito_hogares_yoy',
    'precio_m2_vivienda_yoy_lag4',
    'euribor_12m',
    'ipc_var_anual',
    'brent_yoy',
    'mora_hogares'
]

data_var = df[VARIABLES_VAR].copy()

print(f"\n  Variables del sistema VAR: {len(VARIABLES_VAR)}")
for i, var in enumerate(VARIABLES_VAR, 1):
    print(f"    {i}. {var}")

# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE ESTACIONARIEDAD (ADF)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PASO 1: ANÁLISIS DE ESTACIONARIEDAD (ADF Test)")
print("="*80)

adf_resultados = []
for var in VARIABLES_VAR:
    result = adfuller(data_var[var], autolag='AIC')
    estacionaria = result[1] < 0.05
    adf_resultados.append({
        'Variable': var,
        'ADF Stat': round(result[0], 4),
        'P-value': round(result[1], 4),
        'Estacionaria': 'Si' if estacionaria else 'No'
    })
    print(f"  {var:35s} | ADF={result[0]:8.4f} | p={result[1]:.4f} | "
          f"{'Estacionaria' if estacionaria else 'No estacionaria'}")

adf_df = pd.DataFrame(adf_resultados)
no_estacionarias = adf_df[adf_df['Estacionaria'] == 'No']['Variable'].tolist()
print(f"\n  Variables no estacionarias: {len(no_estacionarias)}")
for v in no_estacionarias:
    print(f"    - {v}")
print(f"  Nota: la no estacionariedad limita la interpretacion de las IRF.")

# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO 
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PASO 2: MODELO VAR(1)")
print("="*80)

# Se fija el orden p=1 porque con 81 observaciones y 7 variables endógenas

var_modelo = VAR(data_var)
var_fit = var_modelo.fit(1, ic=None)

print(f"\n  Orden del modelo: VAR(1)")
print(f"  Observaciones: {var_fit.nobs}")
print(f"  Variables endógenas: {var_fit.neqs}")
print(f"  AIC: {var_fit.aic:.4f}")
print(f"  BIC: {var_fit.bic:.4f}")

# Extraer coeficientes de la ecuación de mora_hogares
mora_idx = VARIABLES_VAR.index('mora_hogares')
mora_idx_fevd = var_fit.names.index('mora_hogares')
coef_mora = var_fit.params.iloc[:, mora_idx]

# P-values ecuación mora_hogares
pvalues_mora = var_fit.pvalues.iloc[:, mora_idx]
print(f"\n  P-values ecuación mora_hogares:")
for nombre, pval in pvalues_mora.items():
    signif = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    print(f"    {nombre:35s}: p={pval:.4f} {signif}")

print(f"\n  Coeficientes y p-values ecuación mora_hogares:")
for nombre in coef_mora.index:
    coef = coef_mora[nombre]
    pval = pvalues_mora[nombre]
    signif = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    print(f"    {nombre:35s}: coef={coef:.4f}  p={pval:.4f} {signif}")

# ──────────────────────────────────────────────────────────────────────────────
# TABLA 1: COEFICIENTES VAR 
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PASO 3: TABLAS Y GRÁFICOS")
print("="*80)

def tabla_png(df_tabla, titulo, subtitulo, ruta, col_widths=None):
    n_cols = len(df_tabla.columns)
    n_rows = len(df_tabla)
    alto   = max(2.5, 0.55 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(13, alto))
    ax.axis('off')

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    tabla = ax.table(
        cellText=df_tabla.round(4).astype(str).values,
        colLabels=df_tabla.columns,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 2.2)

    for j in range(n_cols):
        cell = tabla[(0, j)]
        cell.set_facecolor(COLOR_PRINCIPAL)
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('white')

    for i in range(1, n_rows + 1):
        color = 'white' if i % 2 != 0 else '#f5f5f5'
        for j in range(n_cols):
            cell = tabla[(i, j)]
            cell.set_facecolor(color)
            cell.set_edgecolor('#e0e0e0')

    fig.suptitle(titulo, fontsize=12, fontweight='bold', y=0.97)
    ax.set_title(subtitulo, fontsize=9, color='#555555', pad=4)

    plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Tabla guardada: {os.path.basename(ruta)}")

# Tabla coeficientes ecuación mora_hogares
coef_tabla_df = pd.DataFrame({
    'Variable': coef_mora.index,
    'Coeficiente': coef_mora.values.round(4),
    'Interpretacion': [
        'Constante del sistema',
        'Efecto autorregresivo tasa paro (lag 1)',
        'Efecto autorregresivo credito hogares (lag 1)',
        'Efecto autorregresivo precio vivienda (lag 1)',
        'Efecto autorregresivo Euribor (lag 1)',
        'Efecto autorregresivo IPC (lag 1)',
        'Efecto autorregresivo brent (lag 1)',
        'Persistencia temporal de mora (lag 1)'
    ]
})

tabla_png(
    coef_tabla_df,
    'VAR(1) — Coeficientes de la ecuacion de mora_hogares',
    'Efecto de cada variable en t-1 sobre mora_hogares en t',
    os.path.join(RUTA_GRAFICOS, 'tabla_coeficientes_var.png'),
    col_widths=[0.35, 0.20, 0.45]
)

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES IMPULSO-RESPUESTA (IRF) — 6 trimestres
# ──────────────────────────────────────────────────────────────────────────────

irf = var_fit.irf(6)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    'VAR(1) — Funciones de Impulso-Respuesta sobre mora_hogares\n'
    '(impacto de un shock unitario en cada variable, horizonte 6 trimestres)',
    fontsize=12, fontweight='bold'
)

# Solo mostramos el impacto de las 6 variables predictoras sobre mora_hogares
# mora_idx es el índice de mora_hogares en el sistema
for idx, shock_var in enumerate(VARIABLES_VAR[:-1]):  # excluimos mora_hogares como shock
    ax = axes.flatten()[idx]
    shock_idx = VARIABLES_VAR.index(shock_var)

    impulso = irf.irfs[:, mora_idx, shock_idx]
    periodos = range(len(impulso))

    ax.plot(periodos, impulso, 'o-', linewidth=2, markersize=6,
            color=COLOR_PRINCIPAL)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(periodos, impulso, 0,
                    where=[v > 0 for v in impulso],
                    alpha=0.15, color=COLOR_PRINCIPAL)
    ax.fill_between(periodos, impulso, 0,
                    where=[v < 0 for v in impulso],
                    alpha=0.15, color=COLOR_TEST)

    ax.set_xlabel('Trimestres tras el shock', fontsize=9)
    ax.set_ylabel('Impacto sobre mora_hogares (M€)', fontsize=9)
    ax.set_title(f'Shock: {shock_var}', fontsize=10, fontweight='bold')
    ax.set_xticks(range(7))
    ax.grid(alpha=0.3)

plt.tight_layout()
ruta_irf = os.path.join(RUTA_GRAFICOS, 'irf_impacto_en_mora.png')
plt.savefig(ruta_irf, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: irf_impacto_en_mora.png")

# ──────────────────────────────────────────────────────────────────────────────
# FEVD — Descomposición de Varianza del Error
# ──────────────────────────────────────────────────────────────────────────────
# Complementa las IRF mostrando la importancia relativa de cada variable

fevd = var_fit.fevd(6)
fevd_mora = fevd.decomp[mora_idx_fevd, :, :] # shape: (horizonte, n_variables)

# Tabla FEVD
fevd_data = []
for t in range(6):
    row = {'Horizonte': f't+{t+1}'}
    for v_idx, v_name in enumerate(VARIABLES_VAR):
        row[v_name] = round(fevd_mora[t, v_idx] * 100, 2)
    fevd_data.append(row)

fevd_df = pd.DataFrame(fevd_data).set_index('Horizonte')

tabla_png(
    fevd_df.reset_index(),
    'VAR(1) — Descomposicion de Varianza del Error (FEVD)',
    'Contribucion de cada variable a la varianza de mora_hogares (%)',
    os.path.join(RUTA_GRAFICOS, 'tabla_fevd_var.png')
)

# Gráfico FEVD 
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(
    'VAR(1) — Descomposicion de Varianza del Error (FEVD)\n'
    '(contribucion de cada variable a la varianza de mora_hogares por horizonte)',
    fontsize=12, fontweight='bold'
)

colores_vars = [
    '#003366', '#1f77b4', '#4e9a8f', '#d62728', '#ff7f0e', '#8c564b', '#7f7f7f'
]

x_time = np.arange(1, 7)
y_data = [fevd_mora[:, j] * 100 for j in range(len(VARIABLES_VAR))]

ax.stackplot(x_time, y_data, labels=VARIABLES_VAR,
             colors=colores_vars, alpha=0.75)
ax.set_xlabel('Horizonte (trimestres)', fontsize=10)
ax.set_ylabel('Contribucion a la varianza (%)', fontsize=10)
ax.set_xticks(range(1, 7))
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
ruta_fevd = os.path.join(RUTA_GRAFICOS, 'fevd_mora_hogares.png')
plt.savefig(ruta_fevd, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: fevd_mora_hogares.png")

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("RESUMEN FINAL — VAR(1)")
print("="*80)
print(f"  Observaciones      : {len(data_var)} (2005-Q1 a 2025-Q1)")
print(f"  Variables endogenas: {len(VARIABLES_VAR)}")
print(f"  Orden del modelo   : VAR(1)")
print(f"  Horizonte IRF/FEVD : 6 trimestres")
print(f"\n  Coeficiente autorregresivo mora_hogares: "
      f"{coef_mora.get('L1.mora_hogares', 'N/A'):.4f}")
print(f"\n  Variables no estacionarias ({len(no_estacionarias)}):")
for v in no_estacionarias:
    print(f"    - {v}")
print(f"\n  Archivos generados en {RUTA_GRAFICOS}:")
print(f"    tabla_coeficientes_var.png")
print(f"    tabla_fevd_var.png")
print(f"    irf_impacto_en_mora.png")
print(f"    fevd_mora_hogares.png")
print("="*80 + "\n")