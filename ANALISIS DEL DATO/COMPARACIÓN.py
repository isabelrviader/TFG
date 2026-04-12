# -*- coding: utf-8 -*-
"""
ANÁLISIS DEL DATO — COMPARATIVA Y SISTEMA DE ALERTA TEMPRANA

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

RUTA_DATOS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\analisis_del_dato\dataset_modelos.xlsx'
RUTA_GRAFICOS = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\COMPARACION'

os.makedirs(RUTA_GRAFICOS, exist_ok=True)

# Estilo visual
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10

COLOR_PRINCIPAL = '#003366'   # Azul oscuro BdE — OLS y elementos principales
COLOR_RF        = '#1f77b4'   # Azul medio — Random Forest
COLOR_TEST      = '#d62728'   # Rojo — Test y alertas
COLOR_OK        = '#2ca02c'   # Verde — Sin alerta

print("\n" + "="*80)
print("PASO 4: COMPARATIVA Y SISTEMA DE ALERTA TEMPRANA")
print("="*80)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(RUTA_DATOS)
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').reset_index(drop=True)

print(f"\n  Dataset cargado: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(f"  mora_hogares — Min: {df['mora_hogares'].min():.1f} M€  "
      f"Max: {df['mora_hogares'].max():.1f} M€  "
      f"Media: {df['mora_hogares'].mean():.1f} M€")
print(f"  (verificación escala: valores esperados entre 2.777 M€ y 50.874 M€)")

FEATURES = ['tasa_paro_lag3', 'credito_hogares_yoy', 'precio_m2_vivienda_yoy_lag4',
            'euribor_12m', 'ipc_var_anual', 'brent_yoy']
TARGET = 'mora_hogares'

mask = df[FEATURES + [TARGET]].notna().all(axis=1)
X = df.loc[mask, FEATURES].values
y = df.loc[mask, TARGET].values
fechas = df.loc[mask, 'fecha'].values

print(f"  Periodo: {df.loc[mask, 'fecha'].min().date()} a {df.loc[mask, 'fecha'].max().date()}")

# ──────────────────────────────────────────────────────────────────────────────
# TABLA
# ──────────────────────────────────────────────────────────────────────────────

def tabla_png(df_tabla, titulo, subtitulo, ruta, col_widths=None):
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

# ──────────────────────────────────────────────────────────────────────────────
# PARTE 1 — COMPARATIVA OLS vs RF
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PARTE 1: COMPARATIVA OLS vs RANDOM FOREST (Pilar 1)")
print("="*80)

HIPERPARAMETROS_RF = {
    'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 5,
    'min_samples_split': 10, 'random_state': 42, 'n_jobs': -1
}

tscv = TimeSeriesSplit(n_splits=4)

ols_folds = []
rf_folds  = []
y_pred_ols_all, y_pred_rf_all = [], []
y_test_all, fold_labels = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    yp_ols_train = ols.predict(X_train)
    yp_ols_test  = ols.predict(X_test)

    # RF
    rf = RandomForestRegressor(**HIPERPARAMETROS_RF)
    rf.fit(X_train, y_train)
    yp_rf_train = rf.predict(X_train)
    yp_rf_test  = rf.predict(X_test)

    ols_folds.append({
        'Fold': fold,
        'RMSE Train': np.sqrt(mean_squared_error(y_train, yp_ols_train)),
        'RMSE Test':  np.sqrt(mean_squared_error(y_test,  yp_ols_test)),
        'MAE Test':   mean_absolute_error(y_test, yp_ols_test),
        'R2 Train':   r2_score(y_train, yp_ols_train),
        'R2 Test':    r2_score(y_test,  yp_ols_test),
    })
    rf_folds.append({
        'Fold': fold,
        'RMSE Train': np.sqrt(mean_squared_error(y_train, yp_rf_train)),
        'RMSE Test':  np.sqrt(mean_squared_error(y_test,  yp_rf_test)),
        'MAE Test':   mean_absolute_error(y_test, yp_rf_test),
        'R2 Train':   r2_score(y_train, yp_rf_train),
        'R2 Test':    r2_score(y_test,  yp_rf_test),
    })

    y_pred_ols_all.extend(yp_ols_test)
    y_pred_rf_all.extend(yp_rf_test)
    y_test_all.extend(y_test)
    fold_labels.extend([fold] * len(y_test))

    print(f"  Fold {fold}: OLS RMSE={ols_folds[-1]['RMSE Test']:.0f} M€  "
          f"RF RMSE={rf_folds[-1]['RMSE Test']:.0f} M€")

df_ols = pd.DataFrame(ols_folds)
df_rf  = pd.DataFrame(rf_folds)

ols_rmse_test = df_ols['RMSE Test'].mean()
rf_rmse_test  = df_rf['RMSE Test'].mean()
ols_r2_test   = df_ols['R2 Test'].mean()
rf_r2_test    = df_rf['R2 Test'].mean()
ols_r2_train  = df_ols['R2 Train'].mean()
rf_r2_train   = df_rf['R2 Train'].mean()
mejora_rmse   = (ols_rmse_test - rf_rmse_test) / ols_rmse_test * 100

y_pred_ols_all = np.array(y_pred_ols_all)
y_pred_rf_all  = np.array(y_pred_rf_all)
y_test_all     = np.array(y_test_all)
fold_labels    = np.array(fold_labels)

residuos_ols = y_test_all - y_pred_ols_all
residuos_rf  = y_test_all - y_pred_rf_all

print(f"\n  OLS  — RMSE Test: {ols_rmse_test:.0f} M€  R² Test: {ols_r2_test:.4f}")
print(f"  RF   — RMSE Test: {rf_rmse_test:.0f} M€  R² Test: {rf_r2_test:.4f}")
print(f"  Mejora RF vs OLS: {mejora_rmse:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICO 1: RMSE Test por fold 
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("GENERANDO GRÁFICOS Y TABLAS")
print("="*80)

folds = np.arange(1, 5)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(
    'Comparativa OLS vs Random Forest — RMSE Test por fold\n'
    '(validación temporal, TimeSeriesSplit 4 folds)',
    fontsize=12, fontweight='bold'
)

bars_ols = ax.bar(folds - width/2, df_ols['RMSE Test'], width,
                  label='OLS', color=COLOR_PRINCIPAL, alpha=0.75, edgecolor='white')
bars_rf  = ax.bar(folds + width/2, df_rf['RMSE Test'],  width,
                  label='Random Forest', color=COLOR_RF, alpha=0.75, edgecolor='white')

for bar, val in zip(bars_ols, df_ols['RMSE Test']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:.0f}', ha='center', fontsize=9, color=COLOR_PRINCIPAL, fontweight='bold')
for bar, val in zip(bars_rf, df_rf['RMSE Test']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:.0f}', ha='center', fontsize=9, color=COLOR_RF, fontweight='bold')

ax.set_xlabel('Fold', fontsize=10)
ax.set_ylabel('RMSE Test (M€)', fontsize=10)
ax.set_xticks(folds)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, 'rmse_por_fold.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: rmse_por_fold.png")

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICO 2: Real vs Predicho (OLS y RF) 
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Real vs Predicho — OLS vs Random Forest\n'
    '(validación temporal, puntos coloreados por fold)',
    fontsize=12, fontweight='bold'
)

colores_fold = {1: '#e41a1c', 2: '#ff7f00', 3: '#4daf4a', 4: '#377eb8'}

for ax, y_pred, titulo, rmse, r2 in [
    (axes[0], y_pred_ols_all, 'OLS (Benchmark)', ols_rmse_test, ols_r2_test),
    (axes[1], y_pred_rf_all,  'Random Forest (Principal)', rf_rmse_test, rf_r2_test)
]:
    for fold in range(1, 5):
        mask_f = fold_labels == fold
        ax.scatter(y_test_all[mask_f], y_pred[mask_f],
                   label=f'Fold {fold}', alpha=0.7, s=60,
                   color=colores_fold[fold], edgecolors='white', linewidths=0.5)

    lim_min = min(y_test_all.min(), y_pred.min()) * 0.95
    lim_max = max(y_test_all.max(), y_pred.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            '--', color='black', linewidth=1.5, label='Prediccion perfecta')

    ax.set_xlabel('Valores reales (M€)', fontsize=9)
    ax.set_ylabel('Predicciones (M€)', fontsize=9)
    ax.set_title(f'{titulo}\nRMSE={rmse:.0f} M€  |  R²={r2:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, 'real_vs_predicho.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: real_vs_predicho.png")

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICO 3: Residuos OLS vs RF
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Análisis de residuos — OLS vs Random Forest\n'
    '(residuos de predicción en validación temporal)',
    fontsize=12, fontweight='bold'
)

for ax, y_pred, residuos, titulo, color in [
    (axes[0], y_pred_ols_all, residuos_ols, f'OLS  (std = {residuos_ols.std():.0f} M€)', COLOR_PRINCIPAL),
    (axes[1], y_pred_rf_all,  residuos_rf,  f'Random Forest  (std = {residuos_rf.std():.0f} M€)', COLOR_RF)
]:
    ax.scatter(y_pred, residuos, alpha=0.6, s=55,
               color=color, edgecolors='white', linewidths=0.5)
    ax.axhline(0, color=COLOR_TEST, linestyle='--', linewidth=1.5)
    ax.set_xlabel('Prediccion (M€)', fontsize=9)
    ax.set_ylabel('Residuo (M€)', fontsize=9)
    ax.set_title(titulo, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RUTA_GRAFICOS, 'residuos.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Grafico guardado: residuos.png")

# ──────────────────────────────────────────────────────────────────────────────
# TABLA 1: Comparativa de los 4 modelos
# ──────────────────────────────────────────────────────────────────────────────

df_comp = pd.DataFrame({
    'Modelo':        ['OLS', 'Random Forest', 'XGBoost', 'VAR(1)'],
    'R² Train':      [f'{ols_r2_train:.4f}', f'{rf_r2_train:.4f}', '1.0000', 'N/A'],
    'R² Test':       [f'{ols_r2_test:.4f}',  f'{rf_r2_test:.4f}',  '-2.3688', 'N/A'],
    'RMSE Test (M€)':[f'{ols_rmse_test:.0f}', f'{rf_rmse_test:.0f}', '6.388', 'N/A'],
    'Rol':           ['Benchmark regulatorio', 'Modelo principal',
                      'Descartado (sobreajuste)', 'Análisis complementario']
})

tabla_png(
    df_comp,
    'Comparativa de los cuatro modelos',
    'Promedio de 4 folds de validación temporal (TimeSeriesSplit)',
    os.path.join(RUTA_GRAFICOS, 'tabla_comparativa_modelos.png'),
    col_widths=[0.18, 0.14, 0.14, 0.18, 0.36]
)

# ──────────────────────────────────────────────────────────────────────────────
# TABLA 2: RMSE Train y Test por fold para OLS y RF
# ──────────────────────────────────────────────────────────────────────────────

df_folds_tabla = pd.DataFrame({
    'Fold':              [f'Fold {f}' for f in range(1, 5)],
    'OLS RMSE Train':    df_ols['RMSE Train'].round(0).astype(int),
    'OLS RMSE Test':     df_ols['RMSE Test'].round(0).astype(int),
    'RF RMSE Train':     df_rf['RMSE Train'].round(0).astype(int),
    'RF RMSE Test':      df_rf['RMSE Test'].round(0).astype(int),
})

tabla_png(
    df_folds_tabla,
    'RMSE Train y Test por fold — OLS vs Random Forest (M€)',
    'TimeSeriesSplit 4 folds — mora_hogares en M€',
    os.path.join(RUTA_GRAFICOS, 'tabla_metricas_folds.png')
)

# ──────────────────────────────────────────────────────────────────────────────
# PARTE 2 — PILAR 2: INDICADORES EN TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PARTE 2: PILAR 2 — INDICADORES EN TIEMPO REAL")
print("="*80)

# Umbrales históricos: percentil 75 para variables directas (más = más riesgo)
# y percentil 25 para variables inversas (menos = más riesgo)
ultimo = df.iloc[-1]

indicadores = [
    ('tasa_paro_lag3',            'Tasa de paro (lag3)',         'directa',  0.75),
    ('credito_hogares_yoy',       'Credito hogares YoY',         'inversa',  0.25),
    ('euribor_12m',               'Euribor 12M',                 'directa',  0.75),
    ('ipc_var_anual',             'IPC var. anual',              'inversa',  0.25),
    ('precio_m2_vivienda_yoy_lag4','Precio vivienda YoY (lag4)', 'inversa',  0.25),
]

filas_p2 = []
for col, nombre, tipo, percentil in indicadores:
    umbral = df[col].quantile(percentil)
    valor  = ultimo[col]
    if tipo == 'directa':
        alerta = valor > umbral
    else:
        alerta = valor < umbral
    filas_p2.append({
        'Indicador':     nombre,
        'Valor 2025-Q1': round(valor, 2),
        'Umbral (p75/p25)': round(umbral, 2),
        'Alerta':        'SI' if alerta else 'NO'
    })
    print(f"  {nombre:35s}: {valor:.2f} {'>' if tipo=='directa' else '<'} "
          f"{umbral:.2f} → {'ALERTA' if alerta else 'OK'}")

df_p2 = pd.DataFrame(filas_p2)
alertas_activas = (df_p2['Alerta'] == 'SI').sum()
print(f"\n  Alertas activas: {alertas_activas}/5")

# Tabla Pilar 2 con color en columna Alerta
n_cols = len(df_p2.columns)
n_rows = len(df_p2)
fig, ax = plt.subplots(figsize=(13, max(2.5, 0.55 * (n_rows + 2))))
ax.axis('off')

tabla = ax.table(
    cellText=df_p2.values,
    colLabels=df_p2.columns,
    cellLoc='center', loc='center',
    colWidths=[0.40, 0.18, 0.22, 0.20]
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
    color_fila = 'white' if i % 2 != 0 else '#f5f5f5'
    for j in range(n_cols):
        cell = tabla[(i, j)]
        if j == n_cols - 1:  # columna Alerta
            cell.set_facecolor(COLOR_TEST if df_p2.iloc[i-1]['Alerta'] == 'SI' else COLOR_OK)
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor(color_fila)
        cell.set_edgecolor('#e0e0e0')

fig.suptitle('Pilar 2 — Indicadores de estres en tiempo real',
             fontsize=12, fontweight='bold', y=0.97)
ax.set_title(f'Estado a 2025-Q1  ({alertas_activas}/5 alertas activas)',
             fontsize=9, color='#555555', pad=4)

plt.savefig(os.path.join(RUTA_GRAFICOS, 'tabla_indicadores_pilar2.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Tabla guardada: tabla_indicadores_pilar2.png")

# ──────────────────────────────────────────────────────────────────────────────
# PARTE 3 — PILAR 3: ESCENARIOS DE ESTRÉS CON RF + OLS
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PARTE 3: PILAR 3 — ESCENARIOS DE ESTRÉS (RF + OLS)")
print("="*80)

# El Pilar 3 combina RF y OLS para proporcionar un abanico de estimación:
#   - RF: estimación conservadora basada en patrones históricos.
#     No extrapola más allá del rango de entrenamiento, por lo que en shocks
#     pequeños puede subestimar el impacto real.
#   - OLS: sensibilidad marginal regulatoria. Cada punto de paro, euribor o
#     crédito tiene un efecto lineal y exactamente cuantificable. Es el modelo
#     que IFRS 9 y Basilea III exigen para reportar la sensibilidad paramétrica
#     al supervisor. Actúa como benchmark regulatorio del Pilar 3.


# Entrenar ambos modelos en muestra completa
rf_completo = RandomForestRegressor(**HIPERPARAMETROS_RF)
rf_completo.fit(X, y)

ols_completo = LinearRegression()
ols_completo.fit(X, y)

valores_actuales = df[FEATURES].iloc[-1].copy()
mora_base_rf  = rf_completo.predict(valores_actuales.values.reshape(1, -1))[0]
mora_base_ols = ols_completo.predict(valores_actuales.values.reshape(1, -1))[0]

print(f"  Valores actuales (2025-Q1):")
for col in FEATURES:
    print(f"    {col:35s}: {valores_actuales[col]:.2f}")
print(f"  Mora base RF : {mora_base_rf:.0f} M€")
print(f"  Mora base OLS: {mora_base_ols:.0f} M€")

# Definición de escenarios: incrementos sobre valores actuales
escenarios = {
    'Base':     {'tasa_paro_lag3': 0,  'euribor_12m': 0,   'credito_hogares_yoy': 0},
    'Moderado': {'tasa_paro_lag3': +2, 'euribor_12m': +1,  'credito_hogares_yoy': -5},
    'Severo':   {'tasa_paro_lag3': +4, 'euribor_12m': +2,  'credito_hogares_yoy': -10},
    'Extremo':  {'tasa_paro_lag3': +8, 'euribor_12m': +4,  'credito_hogares_yoy': -20},
}

filas_p3 = []
for nombre_esc, cambios in escenarios.items():
    vals = valores_actuales.copy()
    for var, delta in cambios.items():
        vals[var] += delta

    mora_rf  = rf_completo.predict(vals.values.reshape(1, -1))[0]
    mora_ols = ols_completo.predict(vals.values.reshape(1, -1))[0]

    inc_rf_pct  = (mora_rf  / mora_base_rf  - 1) * 100
    inc_ols_pct = (mora_ols / mora_base_ols - 1) * 100

    filas_p3.append({
        'Escenario':          nombre_esc,
        'Paro':    f"+{cambios['tasa_paro_lag3']}",
        'Euribor': f"+{cambios['euribor_12m']}",
        'Credito':  f"{cambios['credito_hogares_yoy']}",
        'Mora RF (M€)':       round(mora_rf, 0),
        'Mora OLS (M€)':      round(mora_ols, 0),
    })
    print(f"  {nombre_esc:10s}: RF={mora_rf:.0f} M€ ({'+' if inc_rf_pct>=0 else ''}{inc_rf_pct:.1f}%)  "
          f"OLS={mora_ols:.0f} M€ ({'+' if inc_ols_pct>=0 else ''}{inc_ols_pct:.1f}%)")

df_p3 = pd.DataFrame(filas_p3)

tabla_png(
    df_p3,
    'Pilar 3 — Escenarios de estres macroeconómico',
    'Horquilla RF (conservador) vs OLS (sensibilidad regulatoria) — valores en M€',
    os.path.join(RUTA_GRAFICOS, 'tabla_escenarios_pilar3.png'),
    col_widths=[0.15, 0.14, 0.16, 0.14, 0.21, 0.20]
)

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("RESUMEN FINAL — COMPARATIVA Y SISTEMA DE ALERTA TEMPRANA")
print("="*80)
print(f"  OLS  RMSE Test promedio : {ols_rmse_test:.0f} M€")
print(f"  RF   RMSE Test promedio : {rf_rmse_test:.0f} M€")
print(f"  Mejora RF vs OLS        : {mejora_rmse:.1f}%")
print(f"  Std residuos OLS        : {residuos_ols.std():.0f} M€")
print(f"  Std residuos RF         : {residuos_rf.std():.0f} M€")
print(f"  Reduccion std residuos  : {(1 - residuos_rf.std()/residuos_ols.std())*100:.0f}%")
print(f"\n  Pilar 2: {alertas_activas}/5 alertas activas a 2025-Q1")
print(f"  Pilar 3 — Escenario base  : RF={mora_base_rf:.0f} M€  OLS={mora_base_ols:.0f} M€")
print(f"  Pilar 3 — Escenario extremo: RF={filas_p3[-1]['Mora RF (M€)']:.0f} M€  "
      f"OLS={filas_p3[-1]['Mora OLS (M€)']:.0f} M€")

print(f"\n  Archivos generados en {RUTA_GRAFICOS}:")
print(f"    rmse_por_fold.png")
print(f"    real_vs_predicho.png")
print(f"    residuos.png")
print(f"    tabla_comparativa_modelos.png")
print(f"    tabla_metricas_folds.png")
print(f"    tabla_indicadores_pilar2.png")
print(f"    tabla_escenarios_pilar3.png")
print("="*80 + "\n")