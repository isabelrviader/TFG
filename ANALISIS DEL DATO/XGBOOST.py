# -*- coding: utf-8 -*-
"""
ANÁLISIS DEL DATO — XGBOOST

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

RUTA_DATOS_ANALISIS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\analisis_del_dato'
RUTA_GRAFICOS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\XGBOOST'

os.makedirs(RUTA_GRAFICOS, exist_ok=True)

# Estilo visual
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams['font.size'] = 10

COLOR_PRINCIPAL = '#003366'   # Azul oscuro BdE — barras y líneas train
COLOR_TEST      = '#d62728'   # Rojo — líneas test
COLOR_NEUTRO    = '#1f77b4'   # Azul medio — scatter

print("\n" + "="*80)
print("PASO 2: XGBOOST (ALTERNATIVA EXPLORADA Y DESCARTADA)")
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

FEATURES = [
    'tasa_paro_lag3',
    'credito_hogares_yoy',
    'precio_m2_vivienda_yoy_lag4',
    'euribor_12m',
    'ipc_var_anual',
    'brent_yoy'
]

X = df[FEATURES].copy()
y = df['mora_hogares'].copy()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE HIPERPARÁMETROS
# ──────────────────────────────────────────────────────────────────────────────

# Hiperparámetros comparables a los del RF para garantizar una evaluación justa:
#   - n_estimators=100, max_depth=4: mismos que RF
#   - learning_rate=0.1: tasa de aprendizaje estándar para series cortas
#   - subsample=0.8: muestreo aleatorio de observaciones por árbol (reduce sobreajuste)
#   - colsample_bytree=0.8: muestreo aleatorio de variables por árbol (reduce sobreajuste)

HIPERPARAMETROS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

print(f"\n  Hiperparámetros XGBoost:")
print(f"    n_estimators   : {HIPERPARAMETROS['n_estimators']} árboles")
print(f"    max_depth      : {HIPERPARAMETROS['max_depth']} niveles")
print(f"    learning_rate  : {HIPERPARAMETROS['learning_rate']}")
print(f"    subsample      : {HIPERPARAMETROS['subsample']}")
print(f"    colsample_bytree: {HIPERPARAMETROS['colsample_bytree']}")

# ──────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN TEMPORAL — TimeSeriesSplit 4 folds
# ──────────────────────────────────────────────────────────────────────────────

# TimeSeriesSplit evita el data leakage al respetar el orden cronológico.

print("\n" + "="*80)
print("PASO 2: VALIDACIÓN TEMPORAL (TimeSeriesSplit, 4 folds)")
print("="*80)

tscv = TimeSeriesSplit(n_splits=4)

resultados_folds = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    xgb = XGBRegressor(**HIPERPARAMETROS)
    xgb.fit(X_train, y_train, verbose=False)

    y_pred_train = xgb.predict(X_train)
    y_pred_test  = xgb.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_train  = mean_absolute_error(y_train, y_pred_train)
    mae_test   = mean_absolute_error(y_test,  y_pred_test)
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)

    print(f"\n  Fold {fold}: Train Obs. {train_idx[0]+1}-{train_idx[-1]+1} "
          f"| Test Obs. {test_idx[0]+1}-{test_idx[-1]+1}")
    print(f"    Train -> RMSE: {rmse_train:.2f} M€   R²: {r2_train:.4f}")
    print(f"    Test  -> RMSE: {rmse_test:.2f} M€   R²: {r2_test:.4f}")

    resultados_folds.append({
        'Fold': fold,
        'N Train': len(train_idx),
        'N Test': len(test_idx),
        'RMSE Train (M€)': rmse_train,
        'RMSE Test (M€)': rmse_test,
        'MAE Train (M€)': mae_train,
        'MAE Test (M€)': mae_test,
        'R² Train': r2_train,
        'R² Test': r2_test,
        '_y_test': y_test.values,
        '_y_pred': y_pred_test,
    })

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN
# ──────────────────────────────────────────────────────────────────────────────

df_cv = pd.DataFrame(resultados_folds).drop_duplicates(subset=['Fold']).reset_index(drop=True)

rmse_test_mean = df_cv['RMSE Test (M€)'].mean()
rmse_test_std  = df_cv['RMSE Test (M€)'].std()
mae_test_mean  = df_cv['MAE Test (M€)'].mean()
r2_train_mean  = df_cv['R² Train'].mean()
r2_test_mean   = df_cv['R² Test'].mean()

print(f"\n  Resumen agregado (promedio 4 folds):")
print(f"    RMSE Test : {rmse_test_mean:.2f} M€  ±  {rmse_test_std:.2f} M€")
print(f"    MAE Test  : {mae_test_mean:.2f} M€")
print(f"    R² Train  : {r2_train_mean:.4f}")
print(f"    R² Test   : {r2_test_mean:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

# Mismo criterio que RF, sobre la muestra completa

xgb_completo = XGBRegressor(**HIPERPARAMETROS)
xgb_completo.fit(X, y, verbose=False)

importancias = xgb_completo.feature_importances_

df_importance = pd.DataFrame({
    'Variable': FEATURES,
    'Importancia (%)': importancias / importancias.sum() * 100,
}).sort_values('Importancia (%)', ascending=False).reset_index(drop=True)

print(f"\n  Feature Importance (muestra completa, mismos hiperparámetros del CV):")
for _, row in df_importance.iterrows():
    print(f"    {row['Variable']:35s}: {row['Importancia (%)']:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# TABLAS
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PASO 3: GENERACIÓN DE TABLAS Y GRÁFICOS")
print("="*80)

def tabla_png(df_tabla, titulo, subtitulo, ruta, col_widths=None):
    """
    Genera tabla profesional en PNG con el mismo estilo visual que OLS y RF.
    Cabecera oscura (#003366), filas alternas blanco/#f5f5f5, sin bordes gruesos.
    """
    n_cols = len(df_tabla.columns)
    n_rows = len(df_tabla)
    alto   = max(2.5, 0.55 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(13, alto))
    ax.axis('off')

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    tabla = ax.table(
        cellText=df_tabla.round(2).astype(str).values,
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

# Tabla CV
cols_tabla_cv = ['Fold', 'N Train', 'N Test',
                 'RMSE Train (M€)', 'RMSE Test (M€)',
                 'MAE Train (M€)', 'MAE Test (M€)',
                 'R² Train', 'R² Test']
tabla_png(
    df_cv[cols_tabla_cv],
    'XGBoost — Métricas de validación temporal por fold',
    'TimeSeriesSplit 4 folds — mora_hogares en M€',
    os.path.join(RUTA_GRAFICOS, 'tabla_cv_xgb.png')
)

# Tabla Feature Importance
tabla_png(
    df_importance[['Variable', 'Importancia (%)']],
    'XGBoost — Feature Importance',
    'Modelo entrenado en muestra completa — reducción de impureza de Gini',
    os.path.join(RUTA_GRAFICOS, 'tabla_feature_importance_xgb.png'),
    col_widths=[0.60, 0.40]
)

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICO 1: FEATURE IMPORTANCE 
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    df_importance['Variable'],
    df_importance['Importancia (%)'],
    color=COLOR_PRINCIPAL,
    alpha=0.75,
    edgecolor='white'
)

for i, (_, row) in enumerate(df_importance.iterrows()):
    ax.text(row['Importancia (%)'] + 0.5, i,
            f"{row['Importancia (%)']:.1f}%",
            va='center', fontsize=9, color='#333333')

ax.set_xlabel('Importancia relativa (% reducción impureza de Gini)', fontsize=10)
ax.set_title(
    'XGBoost — Feature Importance (muestra completa)\n'
    '(contribución de cada variable a la reducción de impureza de Gini)',
    fontsize=12, fontweight='bold'
)
ax.set_xlim(0, df_importance['Importancia (%)'].max() + 8)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
ruta_fi = os.path.join(RUTA_GRAFICOS, 'feature_importance_xgb.png')
plt.savefig(ruta_fi, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: feature_importance_xgb.png")

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICO 2: REAL vs PREDICHO por fold
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    'XGBoost — Real vs Predicho por fold\n'
    '(validación temporal, TimeSeriesSplit 4 folds)',
    fontsize=12, fontweight='bold'
)

for i, ax in enumerate(axes.flatten()):
    fold_data = resultados_folds[i]
    y_test = fold_data['_y_test']
    y_pred = fold_data['_y_pred']
    fold   = fold_data['Fold']
    rmse   = fold_data['RMSE Test (M€)']
    r2     = fold_data['R² Test']

    ax.scatter(y_test, y_pred, alpha=0.65, s=55,
               color=COLOR_NEUTRO, edgecolors='white', linewidths=0.5)

    lim_min = min(y_test.min(), y_pred.min()) * 0.95
    lim_max = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            '--', color=COLOR_TEST, linewidth=1.5, label='Predicción perfecta')

    ax.set_xlabel('Valores reales (M€)', fontsize=9)
    ax.set_ylabel('Predicciones (M€)', fontsize=9)
    ax.set_title(f'Fold {fold}  |  RMSE={rmse:.0f} M€  |  R²={r2:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
ruta_pred = os.path.join(RUTA_GRAFICOS, 'predicciones_xgb_folds.png')
plt.savefig(ruta_pred, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: predicciones_xgb_folds.png")

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICO 3: EVOLUCIÓN DE MÉTRICAS POR FOLD (RMSE, MAE, R²)
# ──────────────────────────────────────────────────────────────────────────────

folds = df_cv['Fold'].values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'XGBoost — Evolución de métricas por fold (validación temporal)\n'
    '(sobreajuste severo: R² Train próximo a 1 en todos los folds)',
    fontsize=12, fontweight='bold'
)

# RMSE
axes[0].plot(folds, df_cv['RMSE Train (M€)'], 'o-', color=COLOR_PRINCIPAL,
             linewidth=2, markersize=7, label='Train')
axes[0].plot(folds, df_cv['RMSE Test (M€)'], 's-', color=COLOR_TEST,
             linewidth=2, markersize=7, label='Test')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('RMSE (M€)')
axes[0].set_title('RMSE por fold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xticks(folds)

# MAE
axes[1].plot(folds, df_cv['MAE Train (M€)'], 'o-', color=COLOR_PRINCIPAL,
             linewidth=2, markersize=7, label='Train')
axes[1].plot(folds, df_cv['MAE Test (M€)'], 's-', color=COLOR_TEST,
             linewidth=2, markersize=7, label='Test')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('MAE (M€)')
axes[1].set_title('MAE por fold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xticks(folds)

# R²
axes[2].plot(folds, df_cv['R² Train'], 'o-', color=COLOR_PRINCIPAL,
             linewidth=2, markersize=7, label='Train')
axes[2].plot(folds, df_cv['R² Test'], 's-', color=COLOR_TEST,
             linewidth=2, markersize=7, label='Test')
axes[2].axhline(0, color='black', linestyle='--', linewidth=1,
                label='R² = 0 (media histórica)')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('R²')
axes[2].set_title('R² por fold')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)
axes[2].set_xticks(folds)

plt.tight_layout()
ruta_metricas = os.path.join(RUTA_GRAFICOS, 'cv_metricas_xgb.png')
plt.savefig(ruta_metricas, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: cv_metricas_xgb.png")

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("RESUMEN FINAL — XGBOOST")
print("="*80)
print(f"  Observaciones      : {len(X)} (2005-Q1 a 2025-Q1)")
print(f"  Variables          : {len(FEATURES)} predictoras + mora_hogares (M€)")
print(f"  RMSE Test promedio : {rmse_test_mean:.2f} M€")
print(f"  R² Test promedio   : {r2_test_mean:.4f}")
print(f"  Variable principal : {df_importance.iloc[0]['Variable']} "
      f"({df_importance.iloc[0]['Importancia (%)']:.1f}%)")
print(f"\n  Archivos generados en {RUTA_GRAFICOS}:")
print(f"    tabla_cv_xgb.png")
print(f"    tabla_feature_importance_xgb.png")
print(f"    feature_importance_xgb.png")
print(f"    predicciones_xgb_folds.png")
print(f"    cv_metricas_xgb.png")
print("="*80 + "\n")