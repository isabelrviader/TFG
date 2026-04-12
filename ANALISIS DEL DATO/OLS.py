# -*- coding: utf-8 -*-
"""
ANÁLISIS DEL DATO — MODELO OLS 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

RUTA_DATOS_BRUTOS   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\transformados\dataset_final.xlsx'
RUTA_DATOS_ANALISIS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\analisis_del_dato'
RUTA_GRAFICOS_OLS   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\OLS'

Path(RUTA_DATOS_ANALISIS).mkdir(parents=True, exist_ok=True)
Path(RUTA_GRAFICOS_OLS).mkdir(parents=True, exist_ok=True)

# Estilo visual 
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams['font.size'] = 10

# Paleta de colores 
COLOR_OLS  = '#1f77b4'   # azul — modelo OLS
COLOR_NEG  = '#d62728'   # rojo — efectos negativos / test
COLOR_POS  = '#003366'   # azul oscuro — efectos positivos


# ──────────────────────────────────────────────────────────────────────────────
# PASO 0: CARGA Y PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  PASO 0 — CARGA Y PREPARACIÓN DE DATOS")
print("="*65)

df = pd.read_excel(RUTA_DATOS_BRUTOS)
print(f"\n  Dataset cargado: {df.shape[0]} observaciones, {df.shape[1]} variables")

# Crear lags si no existen en el dataset (tasa_paro_lag3 y precio_m2_vivienda_yoy_lag4
# no están en el dataset_final porque se seleccionaron en el EDA después de la unión)
if 'tasa_paro_lag3' not in df.columns:
    df['tasa_paro_lag3'] = df['tasa_paro'].shift(3)
    print("  tasa_paro_lag3 calculada (shift 3 trimestres)")

if 'precio_m2_vivienda_yoy_lag4' not in df.columns:
    df['precio_m2_vivienda_yoy_lag4'] = df['precio_m2_vivienda_yoy'].shift(4)
    print("  precio_m2_vivienda_yoy_lag4 calculada (shift 4 trimestres)")

# Variables predictoras seleccionadas en el EDA mediante cross-correlation y VIF
X = df[[
    'tasa_paro_lag3',
    'credito_hogares_yoy',
    'precio_m2_vivienda_yoy_lag4',
    'euribor_12m',
    'ipc_var_anual',
    'brent_yoy'
]].copy()

# Variable objetivo: mora_hogares viene en miles de euros en be0413
# Se divide entre 1.000 para expresarla en millones de euros (M€)
y = df['mora_hogares'].copy() / 1_000

print(f"\n  mora_hogares escalada a M€ (÷ 1.000)")
print(f"  Mínimo: {y.min():.1f} M€  |  Máximo: {y.max():.1f} M€  |  Media: {y.mean():.1f} M€")

# Eliminar filas con NaN generados por los lags (primeros trimestres sin historia)
mask_validas = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask_validas].reset_index(drop=True)
y_clean = y[mask_validas].reset_index(drop=True)

print(f"\n  Observaciones válidas tras eliminar NaN de lags: {len(X_clean)}")
print(f"  Periodo: 2005-Q1 a 2025-Q1")

# Guardar dataset modelo para que RF, XGBoost y VAR usen exactamente los mismos datos
if 'fecha' in df.columns:
    df_modelo = pd.concat([
        df[mask_validas]['fecha'].reset_index(drop=True),
        X_clean,
        pd.Series(y_clean, name='mora_hogares')
    ], axis=1)
else:
    df_modelo = pd.concat([X_clean, pd.Series(y_clean, name='mora_hogares')], axis=1)

ruta_modelo = os.path.join(RUTA_DATOS_ANALISIS, 'dataset_modelos.xlsx')
df_modelo.to_excel(ruta_modelo, index=False)
print(f"\n  dataset_modelos.xlsx guardado: {len(df_modelo)} obs, mora en M€")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1: ESTIMACIÓN OLS SOBRE MUESTRA COMPLETA
#
# Se estima el modelo sobre las 81 observaciones completas para obtener:
# · Coeficientes en muestra y su significatividad estadística
# · Diagnósticos de supuestos (normalidad, homocedasticidad, autocorrelación)
# El OLS no se evalúa por su capacidad predictiva fuera de muestra (eso lo
# hace el paso 2) sino por la interpretabilidad de sus parámetros.
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  PASO 1 — OLS EN MUESTRA COMPLETA")
print("="*65)

# Statsmodels requiere añadir la constante manualmente
X_const   = sm.add_constant(X_clean)
modelo_ols = sm.OLS(y_clean, X_const).fit()

print(f"\n  R² ajustado    : {modelo_ols.rsquared_adj:.4f}")
print(f"  F-estadístico  : {modelo_ols.fvalue:.2f}  (p < 0.001)")
print(f"  AIC            : {modelo_ols.aic:.2f}")
print(f"  Durbin-Watson  : {sm.stats.durbin_watson(modelo_ols.resid):.4f}")

# Tabla de coeficientes
coef_df = pd.DataFrame({
    'Variable'    : modelo_ols.params.index,
    'Coeficiente' : modelo_ols.params.values,
    'Std. Error'  : modelo_ols.bse.values,
    't-Stat'      : modelo_ols.tvalues.values,
    'P-value'     : modelo_ols.pvalues.values,
    'Signif.'     : ['***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else '' for p in modelo_ols.pvalues]
})

print(f"\n  Coeficientes estimados:")
print(f"  {'Variable':<35} {'Coef':>10} {'Std.Err':>10} {'t-Stat':>8} {'P-value':>8} {'Signif.':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
for _, row in coef_df.iterrows():
    print(f"  {row['Variable']:<35} {row['Coeficiente']:>10.4f} {row['Std. Error']:>10.4f} "
          f"{row['t-Stat']:>8.3f} {row['P-value']:>8.4f} {row['Signif.']:>8}")

# --- Tabla de coeficientes ---
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.axis('off')
fig.suptitle("OLS — Tabla de coeficientes estimados\n"
             "(variable objetivo: mora_hogares en M€, muestra completa 2005-2025)",
             fontsize=12, fontweight='bold', y=1.02)

coef_sin_const = coef_df[coef_df['Variable'] != 'const'].copy()
tabla_datos = []
for _, row in coef_sin_const.iterrows():
    tabla_datos.append([
        row['Variable'],
        f"{row['Coeficiente']:.4f}",
        f"{row['Std. Error']:.4f}",
        f"{row['t-Stat']:.3f}",
        f"{row['P-value']:.4f}",
        row['Signif.']
    ])

cabeceras = ['Variable', 'Coeficiente (M€)', 'Std. Error', 't-Stat', 'P-value', 'Signif.']
tabla = ax.table(cellText=tabla_datos, colLabels=cabeceras,
                 cellLoc='center', loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 2.2)

# Estilo 
for j in range(len(cabeceras)):
    tabla[0, j].set_facecolor('#212121')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

# Filas alternadas y color según significatividad
for i in range(len(tabla_datos)):
    color_fila = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
    for j in range(len(cabeceras)):
        tabla[i+1, j].set_facecolor(color_fila)
        tabla[i+1, j].set_linewidth(0.5)

plt.tight_layout()
ruta = os.path.join(RUTA_GRAFICOS_OLS, 'tabla_coeficientes_ols.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  tabla_coeficientes_ols.png guardada")

# --- Gráfico de coeficientes (barras horizontales) ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("OLS — Coeficientes estimados\n"
             "(efecto marginal de cada predictor sobre mora_hogares en M€)",
             fontsize=13, fontweight='bold')

coef_plot = coef_sin_const.sort_values('Coeficiente')
colores   = [COLOR_POS if x > 0 else COLOR_NEG for x in coef_plot['Coeficiente']]

ax.barh(range(len(coef_plot)), coef_plot['Coeficiente'], color=colores, alpha=0.8,
        edgecolor='white')
ax.set_yticks(range(len(coef_plot)))
ax.set_yticklabels(coef_plot['Variable'], fontsize=10)
ax.set_xlabel('Coeficiente (M€ por unidad del predictor)', fontsize=10)
ax.axvline(x=0, color='black', linewidth=0.8)

for i, (_, row) in enumerate(coef_plot.iterrows()):
    offset = 0.01 if row['Coeficiente'] >= 0 else -0.01
    ha     = 'left' if row['Coeficiente'] >= 0 else 'right'
    ax.text(row['Coeficiente'] + offset, i,
            f"{row['Coeficiente']:.4f} {row['Signif.']}",
            va='center', fontsize=9)

leyenda = [mpatches.Patch(color=COLOR_POS, label='Efecto positivo'),
           mpatches.Patch(color=COLOR_NEG, label='Efecto negativo')]
ax.legend(handles=leyenda, fontsize=9, loc='lower right')

plt.tight_layout()
ruta = os.path.join(RUTA_GRAFICOS_OLS, 'coeficientes_ols.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  coeficientes_ols.png guardado")

# --- Gráficos diagnósticos de residuos ---
# Se comprueban los tres supuestos del OLS: normalidad, homocedasticidad
# y ausencia de autocorrelación. 
residuals = modelo_ols.resid
fitted    = modelo_ols.fittedvalues
dw        = sm.stats.durbin_watson(residuals)
_, shap_p = shapiro(residuals)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("OLS — Diagnósticos de residuos en muestra completa\n"
             f"(Durbin-Watson: {dw:.3f}  |  Shapiro-Wilk p-value: {shap_p:.4f})",
             fontsize=13, fontweight='bold')

# Panel 1: Residuos vs Valores ajustados — detecta heterocedasticidad y no linealidad
axes[0, 0].scatter(fitted, residuals, alpha=0.6, color=COLOR_OLS, s=40)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_xlabel('Valores ajustados (M€)')
axes[0, 0].set_ylabel('Residuos (M€)')
axes[0, 0].set_title('Residuos vs Valores ajustados')

# Panel 2: Q-Q plot — detecta desviaciones de la normalidad en las colas
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (normalidad de residuos)')

# Panel 3: Distribución de residuos — confirma asimetría y colas pesadas
axes[1, 0].hist(residuals, bins=15, color=COLOR_OLS, alpha=0.7, edgecolor='white')
axes[1, 0].set_xlabel('Residuos (M€)')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].set_title('Distribución de residuos')

# Panel 4: Scale-Location — detecta heterocedasticidad (varianza no constante)
std_resid = residuals / residuals.std()
axes[1, 1].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.6, color=COLOR_OLS, s=40)
axes[1, 1].set_xlabel('Valores ajustados (M€)')
axes[1, 1].set_ylabel('√|Residuos estandarizados|')
axes[1, 1].set_title('Scale-Location (homocedasticidad)')

plt.tight_layout()
ruta = os.path.join(RUTA_GRAFICOS_OLS, 'diagnostico_ols.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  diagnostico_ols.png guardado")
print(f"\n  Durbin-Watson: {dw:.4f}  (valor ideal = 2, indica autocorrelación positiva severa)")
print(f"  Shapiro-Wilk p: {shap_p:.4f}  (< 0.05 rechaza normalidad de residuos)")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2: VALIDACIÓN TEMPORAL CON TIMESERIESSPLIT
# Se usan 4 folds para tener suficientes observaciones en cada test
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  PASO 2 — VALIDACIÓN TEMPORAL (TimeSeriesSplit, 4 folds)")
print("="*65)

tscv          = TimeSeriesSplit(n_splits=4)
resultados_cv = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean), 1):

    X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

    # Entrenar OLS en el fold de entrenamiento
    X_tr_c  = sm.add_constant(X_train)
    mod_fold = sm.OLS(y_train, X_tr_c).fit()

    # Predecir sobre train y test
    y_tr_pred  = mod_fold.predict(X_tr_c)
    X_te_c     = sm.add_constant(X_test)
    y_te_pred  = mod_fold.predict(X_te_c)

    # Métricas: RMSE, MAE y R²
    rmse_tr = np.sqrt(np.mean((y_train - y_tr_pred) ** 2))
    rmse_te = np.sqrt(np.mean((y_test  - y_te_pred) ** 2))
    mae_tr  = np.mean(np.abs(y_train - y_tr_pred))
    mae_te  = np.mean(np.abs(y_test  - y_te_pred))
    r2_tr   = 1 - np.sum((y_train - y_tr_pred)**2) / np.sum((y_train - y_train.mean())**2)
    r2_te   = 1 - np.sum((y_test  - y_te_pred)**2) / np.sum((y_test  - y_test.mean())**2)

    print(f"\n  Fold {fold}:  Train {train_idx[0]+1}-{train_idx[-1]+1} obs  |  "
          f"Test {test_idx[0]+1}-{test_idx[-1]+1} obs")
    print(f"    Train → RMSE: {rmse_tr:.2f} M€   R²: {r2_tr:.4f}")
    print(f"    Test  → RMSE: {rmse_te:.2f} M€   R²: {r2_te:.4f}")

    resultados_cv.append({
        'Fold': fold,
        'N_train': len(train_idx), 'N_test': len(test_idx),
        'RMSE_train': rmse_tr, 'RMSE_test': rmse_te,
        'MAE_train': mae_tr,   'MAE_test': mae_te,
        'R2_train': r2_tr,     'R2_test': r2_te
    })

df_cv = pd.DataFrame(resultados_cv)

# Promedios y desviaciones estándar para el resumen agregado
rmse_te_mean = df_cv['RMSE_test'].mean()
rmse_te_std  = df_cv['RMSE_test'].std()
mae_te_mean  = df_cv['MAE_test'].mean()
r2_te_mean   = df_cv['R2_test'].mean()
r2_tr_mean   = df_cv['R2_train'].mean()

print(f"\n  Resumen agregado (promedio 4 folds):")
print(f"    RMSE Test : {rmse_te_mean:.2f} M€  ±  {rmse_te_std:.2f} M€")
print(f"    MAE Test  : {mae_te_mean:.2f} M€")
print(f"    R² Train  : {r2_tr_mean:.4f}")
print(f"    R² Test   : {r2_te_mean:.4f}")

# --- Tabla visual de métricas por fold ---
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis('off')
fig.suptitle("OLS — Métricas de validación temporal por fold\n"
             "(TimeSeriesSplit 4 folds — mora_hogares en M€)",
             fontsize=12, fontweight='bold', y=1.02)

tabla_datos = []
for _, row in df_cv.iterrows():
    tabla_datos.append([
        f"Fold {int(row['Fold'])}",
        f"{int(row['N_train'])}",
        f"{int(row['N_test'])}",
        f"{row['RMSE_train']:.2f}",
        f"{row['RMSE_test']:.2f}",
        f"{row['MAE_train']:.2f}",
        f"{row['MAE_test']:.2f}",
        f"{row['R2_train']:.4f}",
        f"{row['R2_test']:.4f}"
    ])

cabeceras = ['Fold', 'N Train', 'N Test',
             'RMSE Train (M€)', 'RMSE Test (M€)',
             'MAE Train (M€)',  'MAE Test (M€)',
             'R² Train', 'R² Test']

tabla = ax.table(cellText=tabla_datos, colLabels=cabeceras,
                 cellLoc='center', loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(8.5)
tabla.scale(1, 2.2)

for j in range(len(cabeceras)):
    tabla[0, j].set_facecolor('#212121')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

for i in range(len(tabla_datos)):
    color_fila = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
    for j in range(len(cabeceras)):
        tabla[i+1, j].set_facecolor(color_fila)
        tabla[i+1, j].set_linewidth(0.5)

plt.tight_layout()
ruta = os.path.join(RUTA_GRAFICOS_OLS, 'tabla_cv_ols.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  tabla_cv_ols.png guardada")

# --- Gráficos de métricas por fold ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("OLS — Evolución de métricas por fold (validación temporal)\n"
             "(tendencia decreciente del error conforme crece el conjunto de entrenamiento)",
             fontsize=13, fontweight='bold')

folds = df_cv['Fold']

# RMSE Train vs Test
axes[0].plot(folds, df_cv['RMSE_train'], marker='o', color=COLOR_OLS,
             linewidth=2, label='Train')
axes[0].plot(folds, df_cv['RMSE_test'],  marker='s', color=COLOR_NEG,
             linewidth=2, label='Test')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('RMSE (M€)')
axes[0].set_title('RMSE por fold')
axes[0].legend()
axes[0].set_xticks(folds)

# MAE Train vs Test
axes[1].plot(folds, df_cv['MAE_train'], marker='o', color=COLOR_OLS,
             linewidth=2, label='Train')
axes[1].plot(folds, df_cv['MAE_test'],  marker='s', color=COLOR_NEG,
             linewidth=2, label='Test')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('MAE (M€)')
axes[1].set_title('MAE por fold')
axes[1].legend()
axes[1].set_xticks(folds)

# R² Train vs Test
axes[2].plot(folds, df_cv['R2_train'], marker='o', color=COLOR_OLS,
             linewidth=2, label='Train')
axes[2].plot(folds, df_cv['R2_test'],  marker='s', color=COLOR_NEG,
             linewidth=2, label='Test')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6,
                label='R² = 0 (media histórica)')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('R²')
axes[2].set_title('R² por fold')
axes[2].legend()
axes[2].set_xticks(folds)

plt.tight_layout()
ruta = os.path.join(RUTA_GRAFICOS_OLS, 'cv_metricas_ols.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  cv_metricas_ols.png guardado")


# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  RESUMEN FINAL — OLS")
print("="*65)
print(f"\n  Observaciones      : {len(X_clean)} (2005-Q1 a 2025-Q1)")
print(f"  Variables          : 6 predictoras + mora_hogares (M€)")
print(f"  R² ajustado        : {modelo_ols.rsquared_adj:.4f}")
print(f"  F-estadístico      : {modelo_ols.fvalue:.2f} (p < 0.001)")
print(f"  Durbin-Watson      : {sm.stats.durbin_watson(modelo_ols.resid):.4f}")
print(f"  RMSE Test promedio : {rmse_te_mean:.2f} M€")
print(f"  R² Test promedio   : {r2_te_mean:.4f}")
print(f"\n  Archivos generados en {RUTA_GRAFICOS_OLS}:")
print(f"    tabla_coeficientes_ols.png")
print(f"    coeficientes_ols.png")
print(f"    diagnostico_ols.png")
print(f"    tabla_cv_ols.png")
print(f"    cv_metricas_ols.png")
print("="*65 + "\n")