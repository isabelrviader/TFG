# -*- coding: utf-8 -*-
"""
EXPLORACIÓN Y LIMPIEZA — BLOQUE BdE (Banco de España)        

Fuentes:                                                            
· be0413 → Créditos y dudosos por finalidades (EC y EFC)            
· be1901 → Tipos de interés oficiales y de referencia               
· be2507 → Estadísticas generales: vivienda y construcción

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

FILES_BDE = {
    "be0413": r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\be0413.xlsx',
    "be1901": r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\be1901.xlsx',
    "be2507": r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\be2507.xlsx',
}

OUTPUT_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\limpios'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\BDE'

SKIPROWS = 6  # Las primeras 6 filas de cada archivo BdE son metadatos

os.makedirs(OUTPUT_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)

# Estilo visual
plt.style.use("seaborn-v0_8-whitegrid")
COLORES = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: RESUMEN DE VARIABLES Y OBSERVACIONES
# ──────────────────────────────────────────────────────────────────────────────

def resumen_dataset(nombre, df):
    """
    Imprime nº observaciones, nº variables, dtype, % nulos y rango temporal.
    Útil tanto para el dataset raw como para el limpio.
    """
    print(f"\n{'='*65}")
    print(f"  ARCHIVO: {nombre}")
    print(f"{'='*65}")
    print(f"  Observaciones (filas) : {df.shape[0]}")
    print(f"  Variables (columnas)  : {df.shape[1]}")
    print(f"\n  Columnas:")
    for i, col in enumerate(df.columns):
        serie   = df.iloc[:, i]
        n_nulos = int(serie.isna().sum())
        pct     = round(n_nulos / len(df) * 100, 1)
        print(f"    · {str(col):<40} dtype: {str(serie.dtype):<10} "
              f"nulos: {n_nulos:>4} ({pct:>5}%)")
    print(f"\n  Rango temporal: {df.index[0].date()}  →  {df.index[-1].date()}")
    print(f"{'='*65}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: CARGA RAW — ARCHIVOS BdE
# ──────────────────────────────────────────────────────────────────────────────

def cargar_bde_raw(filepath):
    """
    Carga un archivo BdE respetando su estructura:
      · Filas 0-5: metadatos (código, alias, descripción, unidades,
                   frecuencia) → se saltan con skiprows=6
      · Fila 6+  : datos. Columna 0 = fecha 'MMM YYYY' (ej: 'ENE 2000')
      · Guiones '-': valor ausente BdE → se convierten a NaN
    El índice final es DatetimeIndex mensual ordenado.
    """
    df = pd.read_excel(filepath, sheet_name=0, header=None, skiprows=SKIPROWS)
    df = df.rename(columns={0: "fecha"})

    # Usar la fila 0 del archivo original como nombres de columna
    headers    = pd.read_excel(filepath, sheet_name=0, header=None, nrows=1)
    df.columns = ["fecha"] + list(headers.iloc[0, 1:])

    # Traducción meses español → inglés para pd.to_datetime
    meses_es = {
        "ENE": "Jan", "FEB": "Feb", "MAR": "Mar", "ABR": "Apr",
        "MAY": "May", "JUN": "Jun", "JUL": "Jul", "AGO": "Aug",
        "SEP": "Sep", "OCT": "Oct", "NOV": "Nov", "DIC": "Dec"
    }
    def parse_fecha(s):
        try:
            for es, en in meses_es.items():
                s = str(s).replace(es, en)
            return pd.to_datetime(s, format="%b %Y")
        except Exception:
            return pd.NaT

    df["fecha"] = df["fecha"].apply(parse_fecha)
    df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()
    df = df.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.1.E — EXPLORACIÓN INICIAL
# Revisión de columnas, formato fechas, frecuencia, nulos, rango
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 1.1.E — EXPLORACIÓN INICIAL DE LOS DATASETS BdE")
print("="*65)

datasets_raw = {}
for nombre, fichero in FILES_BDE.items():
    df = cargar_bde_raw(fichero)
    datasets_raw[nombre] = df
    resumen_dataset(nombre, df)

    print(f"\n  >>> df.head(3):")
    print(df.head(3).to_string())
    print(f"\n  >>> df.tail(3):")
    print(df.tail(3).to_string())
    print(f"\n  >>> df.describe() (raw completo):")
    print(df.describe().round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.2.E — SELECCIÓN DE VARIABLES RELEVANTES Y DETECCIÓN DE OUTLIERS
#
# Variables SELECCIONADAS:
#
#  · credito_hogares (D_MEE62000 — be0413)
#    Crédito total concedido a hogares. Necesario para calcular el ratio
#    de morosidad en la fase de Transformación.
#
#  · credito_empresas (D_MEE61000 — be0413)
#    Crédito total a actividades productivas.
#
#  · mora_hogares (D_MEADU201 — be0413)
#    Créditos dudosos hogares. Variable objetivo del modelo (segmento hogares).
#
#  · mora_empresas (D_MEADU100 — be0413)
#    Créditos dudosos empresas. Variable de referencia (no entra en modelo).
#
#  · euribor_12m (D_1NBAF472 — be1901)
#    Euríbor a 12 meses. Se elige este plazo frente a 1, 3 o 6 meses porque
#    es el índice de revisión mayoritario en hipotecas variables españolas y
#    el que mayor impacto tiene sobre la capacidad de pago de los hogares.
#
#  · precio_m2_vivienda (DHIVTNOAPLPMMUVT_RLI.T — be2507)
#    Precio medio m² vivienda libre tasada, total nacional (€/m²).
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.2.E — SELECCIÓN DE VARIABLES RELEVANTES")
print("="*65)

# Mapeo código BdE → nombre semántico en snake_case
be0413_vars = {
    "D_MEE62000":             "credito_hogares",    # miles €
    "D_MEE61000":             "credito_empresas",   # miles €
    "D_MEADU201":             "mora_hogares",        # miles €
    "D_MEADU100":             "mora_empresas",       # miles €
}
be1901_vars = {
    "D_1NBAF472":             "euribor_12m",         # %
}
be2507_vars = {
    "DHIVTNOAPLPMMUVT_RLI.T": "precio_m2_vivienda",  # €/m²
}

seleccion = {
    "be0413": be0413_vars,
    "be1901": be1901_vars,
    "be2507": be2507_vars,
}

datasets_sel = {}
for nombre, mapping in seleccion.items():
    df       = datasets_raw[nombre]
    cols_ok  = {c: v for c, v in mapping.items() if c in df.columns}
    cols_err = [c for c in mapping if c not in df.columns]
    if cols_err:
        print(f"\n   {nombre}: columnas no encontradas → {cols_err}")
    df_sel = df[list(cols_ok.keys())].rename(columns=cols_ok)
    datasets_sel[nombre] = df_sel
    print(f"\n  {nombre}: {df_sel.shape[1]} variables → {list(df_sel.columns)}")

# ── Detección de outliers: método IQR umbral 3× ───────────────────────────────
# Se usa 3×IQR en lugar del estándar 1.5×IQR porque las series financieras
# tienen ciclos económicos pronunciados: los valores extremos son datos reales, 
# no errores de medición. Se conservan todos los outliers detectados.

print("\n  >>> Detección de outliers (IQR × 3):")
print("  Umbral conservador: valores extremos son eventos económicos reales.")

outliers_info = {}
for nombre, df in datasets_sel.items():
    outliers_info[nombre] = {}
    for col in df.columns:
        s = df[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR    = Q3 - Q1
        mask   = (s < Q1 - 3*IQR) | (s > Q3 + 3*IQR)
        n_out  = mask.sum()
        outliers_info[nombre][col] = n_out
        if n_out > 0:
            print(f"    · {nombre}/{col}: {n_out} outliers → se conservan")
            print(f"      Límites IQR×3: [{round(Q1-3*IQR,2)}, {round(Q3+3*IQR,2)}]")
        else:
            print(f"    · {nombre}/{col}: sin outliers")


# ── GRÁFICO 1 — Histogramas de las variables seleccionadas ────────────────────
# Muestra la distribución y rango de cada variable antes de transformar.
# Justifica paso 1.2.E

# Unimos los tres DataFrames seleccionados para facilitar el plotting
df_sel_all = pd.concat([
    datasets_sel["be0413"],
    datasets_sel["be1901"],
    datasets_sel["be2507"],
], axis=1)

etiquetas = {
    "credito_hogares":    "Crédito Hogares (miles €)",
    "credito_empresas":   "Crédito Empresas (miles €)",
    "mora_hogares":       "Mora Hogares (miles €)",
    "mora_empresas":      "Mora Empresas (miles €)",
    "euribor_12m":        "Euríbor 12m (%)",
    "precio_m2_vivienda": "Precio m² Vivienda (€/m²)",
}
variables = list(df_sel_all.columns)

fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle("BdE — Histogramas + KDE de variables seleccionadas\n"
             "(distribución antes de transformación, datos raw completos)",
             fontsize=13, fontweight="bold")

for ax, col, color in zip(axes.flat, variables, COLORES):
    s = df_sel_all[col].dropna()
    ax.hist(s, bins=30, color=color, alpha=0.55, edgecolor="white",
            density=True, label="Histograma")
    s.plot.kde(ax=ax, color=color, linewidth=2, label="KDE")
    ax.axvline(s.mean(),   color="black", linestyle="--", linewidth=1.2,
               label=f"Media: {s.mean():,.1f}")
    ax.axvline(s.median(), color="gray",  linestyle=":",  linewidth=1.2,
               label=f"Mediana: {s.median():,.1f}")
    ax.set_title(etiquetas[col], fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.legend(fontsize=7)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G1_histogramas_bde.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G1 guardado: {ruta}")


# ── GRÁFICO 2 — Boxplots por variable (detección visual de outliers) ──────────
# Complementa el análisis IQR con visualización de la dispersión,
# mediana, cuartiles y valores extremos por variable.
# Justifica paso 1.2.E

fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle("BdE — Boxplots de variables seleccionadas\n"
             "(detección visual de outliers, datos raw completos)",
             fontsize=13, fontweight="bold")

for ax, col, color in zip(axes.flat, variables, COLORES):
    s = df_sel_all[col].dropna()
    bp = ax.boxplot(s, patch_artist=True, vert=True,
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="red", alpha=0.5))
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.6)
    n_out = outliers_info.get(
        "be0413" if col in be0413_vars.values() else
        "be1901" if col in be1901_vars.values() else "be2507", {}
    ).get(col, 0)
    ax.set_title(f"{etiquetas[col]}\n(outliers IQR×3: {n_out})",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([])

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G2_boxplots_bde.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G2 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.3.E — RENOMBRADO DE VARIABLES
# Todos los códigos técnicos del BdE se sustituyen por nombres
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.3.E — RENOMBRADO DE VARIABLES")
print("="*65)

mapeo_total = {**be0413_vars, **be1901_vars, **be2507_vars}
print(f"\n  {'Archivo':<10} {'Código BdE':<40} {'Nombre final':<25} {'Unidad'}")
print(f"  {'-'*10} {'-'*40} {'-'*25} {'-'*15}")
unidades = {
    "credito_hogares":    "miles €",
    "credito_empresas":   "miles €",
    "mora_hogares":       "miles €",
    "mora_empresas":      "miles €",
    "euribor_12m":        "%",
    "precio_m2_vivienda": "€/m²",
}
for codigo, nombre_var in mapeo_total.items():
    archivo = ("be0413" if codigo in be0413_vars else
               "be1901" if codigo in be1901_vars else "be2507")
    print(f"  {archivo:<10} {codigo:<40} {nombre_var:<25} {unidades[nombre_var]}")

print("\n Renombrado completado.")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1.4.E — TRATAMIENTO DE VALORES FALTANTES
#
# Los archivos BdE tienen índice mensual pero publican con frecuencia
# trimestral: los 2 meses intermedios de cada trimestre vienen como '-' (NaN).
# Estrategia: interpolación temporal (method='time').
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 1.4.E — TRATAMIENTO DE VALORES FALTANTES")
print("="*65)
print("  Método: interpolación temporal (method='time')")
print("  Motivo: los archivos BdE tienen índice mensual pero publican")
print("  trimestralmente → 2 de cada 3 filas son NaN estructurales.")

# Guardamos NaN antes para el gráfico comparativo
nulos_antes_dict   = {}
nulos_despues_dict = {}

datasets_clean = {}
for nombre, df in datasets_sel.items():
    print(f"\n  ── {nombre} ──────────────────────────────────────")

    # 1. Diagnóstico ANTES
    nulos_antes = df.isna().sum()
    nulos_antes_dict[nombre] = nulos_antes.copy()
    print("  NaN antes:")
    if nulos_antes.sum() > 0:
        for col, n in nulos_antes[nulos_antes > 0].items():
            print(f"    · {col}: {n} ({round(n/len(df)*100,1)}%)")
    else:
        print("    Ninguno")

    # 2. Interpolación temporal
    df_interp = df.copy()
    for col in df_interp.columns:
        if df_interp[col].isna().sum() > 0:
            df_interp[col] = df_interp[col].interpolate(
                method="time", limit_direction="both")

    # 3. Diagnóstico DESPUÉS
    nulos_despues = df_interp.isna().sum()
    nulos_despues_dict[nombre] = nulos_despues.copy()
    print("  NaN después de interpolación temporal:")
    print("    Ninguno " if nulos_despues.sum() == 0
          else nulos_despues[nulos_despues > 0].to_string())

    # 4. Verificación duplicados
    dup = df_interp.index.duplicated().sum()
    print(f"  Fechas duplicadas: {dup}", "No hay fechas duplicadas" if dup == 0 else "→ eliminando...")
    if dup > 0:
        df_interp = df_interp[~df_interp.index.duplicated(keep="first")]

    # 5. Resumen df.info() equivalente
    completitud = (1 - df_interp.isna().mean()) * 100
    print(f"\n  df.info() equivalente:")
    print(f"    Shape      : {df_interp.shape}")
    print(f"    Index range: {df_interp.index[0].date()} → {df_interp.index[-1].date()}")
    print(f"    Dtypes     : {dict(df_interp.dtypes.value_counts())}")
    print(f"    Total NaN  : {df_interp.isna().sum().sum()}")
    print(f"    Completitud: {completitud.round(1).to_dict()}")

    datasets_clean[nombre] = df_interp

print("\n\n" + "="*65)
print(" EXPLORACIÓN Y LIMPIEZA BdE COMPLETADA")
print("  datasets_clean contiene: " + ", ".join(datasets_clean.keys()))
print("="*65)


# ── GRÁFICO 3 — NaN antes y después de la interpolación ──────────────────────
# Visualiza cuántos valores faltantes había por variable y cómo quedan a cero
# tras la interpolación. Justifica paso 1.4.E

# Consolidamos los NaN antes y después en un único DataFrame para el plot
nulos_antes_total  = pd.concat(nulos_antes_dict.values())
nulos_despues_total = pd.concat(nulos_despues_dict.values())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("BdE — Valores nulos (NaN) antes y después de interpolación temporal",
             fontsize=13, fontweight="bold")

axes[0].bar(nulos_antes_total.index, nulos_antes_total.values,
            color="#E91E63", alpha=0.8, edgecolor="white")
axes[0].set_title("Antes de interpolación", fontweight="bold", fontsize=11)
axes[0].set_ylabel("Nº de valores nulos")
axes[0].tick_params(axis="x", rotation=35)
for i, v in enumerate(nulos_antes_total.values):
    if v > 0:
        axes[0].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=8)

axes[1].bar(nulos_despues_total.index, nulos_despues_total.values,
            color="#4CAF50", alpha=0.8, edgecolor="white")
axes[1].set_title("Después de interpolación", fontweight="bold", fontsize=11)
axes[1].set_ylabel("Nº de valores nulos")
axes[1].tick_params(axis="x", rotation=35)
axes[1].set_ylim(0, max(nulos_antes_total.values) * 1.1)  # misma escala
for i, v in enumerate(nulos_despues_total.values):
    axes[1].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G3_nulos_antes_despues_bde.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G3 guardado: {ruta}")


# ── GRÁFICO 4 — Completitud del dataset limpio ────────────────────────────────
# Se calcula la completitud DENTRO DEL RANGO PROPIO de cada archivo,
# Esto muestra la validación real de la interpolación.
#
# Lógica:
#   · be0413 tiene 4 variables → se muestran todas
#   · be1901 tiene 1 variable  → completitud dentro de su propio rango
#   · be2507 tiene 1 variable  → completitud dentro de su propio rango
# Las tres deben mostrar 100% si la interpolación funcionó.

completitud_por_archivo = {}
for nombre, df in datasets_clean.items():
    comp = (1 - df.isna().mean()) * 100
    completitud_por_archivo[nombre] = comp

completitud_final = pd.concat(completitud_por_archivo.values())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(completitud_final.index, completitud_final.values,
               color=COLORES[:len(completitud_final)], alpha=0.8,
               edgecolor="white")
ax.set_xlim(0, 110)
ax.set_xlabel("% de datos válidos (completitud)", fontsize=11)
ax.set_title("BdE — Completitud del dataset limpio por variable\n"
             "(100% = ningún valor nulo dentro del rango propio de cada archivo)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, completitud_final.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.axvline(100, color="green", linestyle="--", linewidth=1.2, alpha=0.7,
           label="100% completitud")
ax.legend(fontsize=9)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G4_completitud_bde.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f" G4 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR ARCHIVOS LIMPIOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  GUARDANDO ARCHIVOS LIMPIOS")
print("="*65)

for nombre, df in datasets_clean.items():
    ruta = os.path.join(OUTPUT_PATH, f"{nombre}_limpio.xlsx")
    df.to_excel(ruta)
    print(f"   {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — DATASETS LIMPIOS BdE")
print("="*65)

for nombre, df in datasets_clean.items():
    resumen_dataset(nombre, df)

print("\n" + "="*65)
print("  VARIABLES FINALES DEL BLOQUE BdE")
print("="*65)
print(f"\n  {'Archivo':<10} {'Código BdE':<40} {'Nombre final':<25} {'Unidad'}")
print(f"  {'-'*10} {'-'*40} {'-'*25} {'-'*10}")
for codigo, nombre_var in mapeo_total.items():
    archivo = ("be0413" if codigo in be0413_vars else
               "be1901" if codigo in be1901_vars else "be2507")
    print(f"  {archivo:<10} {codigo:<40} {nombre_var:<25} {unidades[nombre_var]}")

print(f"\n   Pipeline BdE completado.")
print(f"     Archivos limpios  : {OUTPUT_PATH}")
print(f"     Gráficos generados: {GRAFICOS_PATH}")
print(f"       G1 Histogramas + KDE (distribución variables seleccionadas)")
print(f"       G2 Boxplots (detección visual outliers)")
print(f"       G3 NaN antes vs después de interpolación")
print(f"       G4 Completitud del dataset limpio (rango propio por archivo)")