# -*- coding: utf-8 -*-
"""
EXPLORACIÓN Y LIMPIEZA — BLOQUE INE (Instituto Nacional de Estadística)

Fuentes:
· 67198 → Contabilidad Nacional Trimestral. PIB pm.
· 65079 → EPA. Tasas globales de actividad y empleo.
· 50917 → IPC. Tasa de variación nacional y por CCAA.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

FILES_INE = {
    "67198": r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\67198.xlsx',   # PIB trimestral
    "65079": r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\65079.xlsx',   # EPA tasa paro
    "50917": r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\50917.xlsx',   # IPC inflación
}

OUTPUT_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\limpios'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\INE'

os.makedirs(OUTPUT_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)

# Estilo visual 
plt.style.use("seaborn-v0_8-whitegrid")
COLORES = ["#2196F3", "#E91E63", "#4CAF50"]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: RESUMEN DE VARIABLES Y OBSERVACIONES
# ──────────────────────────────────────────────────────────────────────────────

def resumen_dataset(nombre, df):
    """
    Imprime nº observaciones, nº variables, dtype, % nulos y rango temporal.
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
# FUNCIONES DE CARGA — ARCHIVOS INE
#
# Los archivos INE tienen estructura TRANSPUESTA respecto al BdE:
#   · Filas 0-5  : metadatos (título, descripción, unidades, etc.)
#   · Fila 6 ó 7 : fechas (en columnas, formato YYYYTQ o YYYYMNN)
#   · Filas sig. : cada fila es una variable distinta

# ──────────────────────────────────────────────────────────────────────────────

def parse_fecha_ine(s):
    """
    Convierte formato INE a datetime:
      · Trimestral: '2000T1' → 2000-01-01
      · Mensual   : '2000M01' → 2000-01-01
    Devuelve NaT si el formato no es reconocido.
    """
    s = str(s).strip()
    try:
        if re.match(r'^\d{4}T[1-4]$', s):
            anio, trim = s.split('T')
            mes = {'1': '01', '2': '04', '3': '07', '4': '10'}[trim]
            return pd.to_datetime(f"{anio}-{mes}-01")
        elif re.match(r'^\d{4}M\d{2}$', s):
            anio, mes = s.split('M')
            return pd.to_datetime(f"{anio}-{mes}-01")
    except Exception:
        pass
    return pd.NaT


def get_bloques_ine(df_raw):
    """
    Localiza los bloques de columnas en archivos INE que publican varias
    medidas en el mismo fichero.
    Devuelve dict {nombre_bloque: columna_inicio}.
    """
    bloques = {}
    for col_idx in range(1, df_raw.shape[1]):
        val = df_raw.iloc[6, col_idx]
        if pd.notna(val) and str(val).strip() != '':
            bloques[str(val).strip()] = col_idx
    return bloques


def cargar_ine_67198(filepath):
    """
    PIB trimestral (67198) — Índice de volumen encadenado, dato base.

    Estructura del archivo:
      · Fila 6: tipo de dato ('Dato base', 'Variación trimestral', ...)
      · Fila 7: fechas en formato YYYYTQ
      · Fila 8: cabecera de sección ('Datos no ajustados...')
      · Fila 9: PIB pm a precios de mercado (la que nos interesa)

    Se usan los datos del bloque 'Dato base' (índice encadenado Base 2015=100)
    porque permiten comparar el nivel del PIB entre periodos.

    """
    df_raw  = pd.read_excel(filepath, sheet_name=0, header=None)
    bloques = get_bloques_ine(df_raw)

    inicio = bloques['Dato base']
    fin    = bloques['Variación trimestral']

    fechas  = [parse_fecha_ine(v) for v in df_raw.iloc[7, inicio:fin]]
    valores = pd.to_numeric(df_raw.iloc[9, inicio:fin], errors='coerce').values

    df = pd.DataFrame({'pib': valores}, index=fechas)
    df.index.name = 'fecha'
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def cargar_ine_65079(filepath):
    """
    EPA — Tasa de paro (65079).

    Estructura del archivo:
      · Fila 6: fechas en formato YYYYTQ (directamente, sin bloque)
      · Fila 7: Tasa global de actividad
      · Fila 8: Tasa global de empleo

    La tasa de paro se calcula como:
        tasa_paro = (actividad - empleo) / actividad × 100
    que equivale al % de activos que no tienen empleo.

    """
    df_raw    = pd.read_excel(filepath, sheet_name=0, header=None)
    fechas    = [parse_fecha_ine(v) for v in df_raw.iloc[6, 1:]]
    actividad = pd.to_numeric(df_raw.iloc[7, 1:], errors='coerce').values
    empleo    = pd.to_numeric(df_raw.iloc[8, 1:], errors='coerce').values

    df = pd.DataFrame({
        'tasa_actividad': actividad,
        'tasa_empleo':    empleo,
    }, index=fechas)
    df.index.name = 'fecha'
    df = df[df.index.notna()].sort_index()

    # Cálculo de la tasa de paro 
    df['tasa_paro'] = (
        (df['tasa_actividad'] - df['tasa_empleo']) / df['tasa_actividad'] * 100
    ).round(2)

    # Devolvemos solo la variable final — actividad y empleo son intermedias
    return df[['tasa_paro']]


def cargar_ine_50917(filepath):
    """
    IPC — Variación anual nacional (50917).

    Estructura del archivo:
      · Fila 6: tipo de dato ('Variación mensual', 'Variación anual', ...)
      · Fila 7: fechas en formato YYYYMNN
      · Fila 8: Nacional

    Se usa 'Variación anual'
    Se usa solo el dato Nacional (se descarta el desglose por CCAA
    al ser datos de panel).
    """
    df_raw  = pd.read_excel(filepath, sheet_name=0, header=None)
    bloques = get_bloques_ine(df_raw)

    inicio = bloques['Variación anual']
    fin    = bloques['Variación en lo que va de año']

    fechas  = [parse_fecha_ine(v) for v in df_raw.iloc[7, inicio:fin]]
    valores = pd.to_numeric(df_raw.iloc[8, inicio:fin], errors='coerce').values

    df = pd.DataFrame({'ipc_var_anual': valores}, index=fechas)
    df.index.name = 'fecha'
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


# Mapa de loaders por archivo
LOADERS = {
    "67198": cargar_ine_67198,
    "65079": cargar_ine_65079,
    "50917": cargar_ine_50917,
}

# Metadatos de cada variable para los gráficos y la tabla final
VARIABLES_INFO = {
    "67198": {"nombre": "pib",          "etiqueta": "PIB pm (índice vol. encadenado)", "unidad": "Índice base 2015=100", "frecuencia": "Trimestral"},
    "65079": {"nombre": "tasa_paro",    "etiqueta": "Tasa de Paro (%)",               "unidad": "%",                    "frecuencia": "Trimestral"},
    "50917": {"nombre": "ipc_var_anual","etiqueta": "IPC Variación Anual (%)",         "unidad": "%",                    "frecuencia": "Mensual"},
}


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.1.E — LIMPIEZA DE DATASETS
# Eliminación de metadatos, selección de columnas,
# conversión a formato numérico, formato de fechas,
# frecuencia temporal, valores nulos o duplicados, rango
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 2.1.E — LIMPIEZA Y CARGA DE LOS DATASETS INE")
print("="*65)
print("  Los archivos INE tienen estructura transpuesta:")
print("  filas = variables, columnas = periodos.")
print("  Se eliminan las filas de metadatos y se transpone el dataset.")

# ── Exploración del archivo RAW ORIGINAL (antes de cualquier limpieza) ────────
# Revisión de columnas, formato, frecuencia, nulos y rango
# Se muestra el estado inicial tal cual viene del INE para justificar
# las decisiones de limpieza que se aplican a continuación.

print("\n\n  ── ESTADO INICIAL DE LOS ARCHIVOS (raw original) ──────────")

for nombre, fichero in FILES_INE.items():
    df_original = pd.read_excel(fichero, sheet_name=0, header=None)
    print(f"\n  {'='*60}")
    print(f"  ARCHIVO RAW: {nombre}")
    print(f"  {'='*60}")
    print(f"  Shape original       : {df_original.shape[0]} filas × "
          f"{df_original.shape[1]} columnas")
    print(f"  → Filas son variables, columnas son periodos (estructura INE)")
    print(f"  → Nº de periodos disponibles: {df_original.shape[1] - 1}")

    # Mostrar las primeras filas de metadatos
    print(f"\n  Primeras 9 filas (metadatos + cabecera):")
    print(df_original.iloc[:9, :5].to_string())

    # Contar NaN en el archivo raw
    total_celdas = df_original.shape[0] * df_original.shape[1]
    total_nan    = df_original.isna().sum().sum()
    print(f"\n  NaN en archivo raw   : {total_nan} "
          f"({round(total_nan/total_celdas*100,1)}% del total)")
    print(f"  Columnas duplicadas  : {df_original.columns.duplicated().sum()}")
    print(f"  Tipos de dato raw    : {dict(df_original.dtypes.value_counts())}")

# ── Carga limpia (después de eliminar metadatos y transponer) ─────────────────
print("\n\n  ── ARCHIVOS TRAS ELIMINACIÓN DE METADATOS Y LIMPIEZA ──────")

datasets_raw = {}
for nombre, fichero in FILES_INE.items():
    print(f"\n  Cargando {nombre}...")
    df = LOADERS[nombre](fichero)
    datasets_raw[nombre] = df
    resumen_dataset(nombre, df)

    print(f"\n  >>> df.head(3):")
    print(df.head(3).to_string())
    print(f"\n  >>> df.tail(3):")
    print(df.tail(3).to_string())
    print(f"\n  >>> df.describe():")
    print(df.describe().round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.2.E — TRATAMIENTO DE VALORES NULOS Y OUTLIERS
#
# Los archivos INE tienen muy pocos NaN estructurales al estar ya en formato
# limpio. Los outliers se analizan con IQR×3 por las mismas razones que en
# el bloque BdE: los valores extremos son eventos económicos reales.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.2.E — TRATAMIENTO DE VALORES NULOS Y OUTLIERS")
print("="*65)

nulos_antes_dict   = {}
nulos_despues_dict = {}
outliers_info      = {}
datasets_clean     = {}

for nombre, df in datasets_raw.items():
    print(f"\n  ── {nombre} ──────────────────────────────────────────")

    # 1. Diagnóstico NaN ANTES
    nulos_antes = df.isna().sum()
    nulos_antes_dict[nombre] = nulos_antes.copy()
    print("  NaN antes del tratamiento:")
    if nulos_antes.sum() > 0:
        for col, n in nulos_antes[nulos_antes > 0].items():
            print(f"    · {col}: {n} ({round(n/len(df)*100,1)}%)")
    else:
        print("    Ninguno")

    # 2. Interpolación temporal si hay NaN
    df_interp = df.copy()
    for col in df_interp.columns:
        if df_interp[col].isna().sum() > 0:
            df_interp[col] = df_interp[col].interpolate(
                method="time", limit_direction="both")

    # 3. Diagnóstico NaN DESPUÉS
    nulos_despues = df_interp.isna().sum()
    nulos_despues_dict[nombre] = nulos_despues.copy()
    print("  NaN después de tratamiento:")
    print("    Ninguno " if nulos_despues.sum() == 0
          else nulos_despues[nulos_despues > 0].to_string())

    # 4. Duplicados
    dup = df_interp.index.duplicated().sum()
    print(f"  Fechas duplicadas: {dup}", "[OK]" if dup == 0 else "→ eliminando...")
    if dup > 0:
        df_interp = df_interp[~df_interp.index.duplicated(keep="first")]

    # 5. Detección de outliers IQR×3
    # Mismo criterio que BdE: umbral conservador para series macroeconómicas
    outliers_info[nombre] = {}
    print("  Outliers (IQR×3):")
    for col in df_interp.columns:
        s      = df_interp[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR    = Q3 - Q1
        mask   = (s < Q1 - 3*IQR) | (s > Q3 + 3*IQR)
        n_out  = mask.sum()
        outliers_info[nombre][col] = n_out
        if n_out > 0:
            print(f"    · {col}: {n_out} outliers → se conservan (dato real)")
            print(f"      Límites IQR×3: [{round(Q1-3*IQR,4)}, {round(Q3+3*IQR,4)}]")
        else:
            print(f"    · {col}: sin outliers [OK]")

    # 6. Resumen df.info() equivalente
    completitud = (1 - df_interp.isna().mean()) * 100
    print(f"\n  df.info() equivalente:")
    print(f"    Shape      : {df_interp.shape}")
    print(f"    Index range: {df_interp.index[0].date()} → {df_interp.index[-1].date()}")
    print(f"    Dtypes     : {dict(df_interp.dtypes.value_counts())}")
    print(f"    Total NaN  : {df_interp.isna().sum().sum()}")
    print(f"    Completitud: {completitud.round(1).to_dict()}")

    datasets_clean[nombre] = df_interp

print("\n\n" + "="*65)
print("  TRATAMIENTO DE VALORES NULOS COMPLETADO")
print("="*65)


# ── GRÁFICO 1 — Histogramas + KDE de las variables seleccionadas ─────────────
# Muestra la distribución y rango de cada variable.
# Justifica pasos 2.1.E y 2.4.E

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("INE — Histogramas + KDE de variables seleccionadas\n"
             "(distribución antes de transformación, datos raw completos)",
             fontsize=13, fontweight="bold")

for ax, (nombre, info), color in zip(axes, VARIABLES_INFO.items(), COLORES):
    col = info["nombre"]
    s   = datasets_raw[nombre][col].dropna()
    ax.hist(s, bins=30, color=color, alpha=0.55, edgecolor="white",
            density=True, label="Histograma")
    s.plot.kde(ax=ax, color=color, linewidth=2, label="KDE")
    ax.axvline(s.mean(),   color="black", linestyle="--", linewidth=1.2,
               label=f"Media: {s.mean():.2f}")
    ax.axvline(s.median(), color="gray",  linestyle=":",  linewidth=1.2,
               label=f"Mediana: {s.median():.2f}")
    ax.set_title(f"{info['etiqueta']}\n({info['unidad']})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G1_histogramas_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G1 guardado: {ruta}")


# ── GRÁFICO 2 — Boxplots (detección visual de outliers) ──────────────────────
# Muestra dispersión, mediana, cuartiles y valores extremos por variable.
# Justifica paso 2.2.E

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("INE — Boxplots de variables seleccionadas\n"
             "(detección visual de outliers, datos raw completos)",
             fontsize=13, fontweight="bold")

for ax, (nombre, info), color in zip(axes, VARIABLES_INFO.items(), COLORES):
    col   = info["nombre"]
    s     = datasets_raw[nombre][col].dropna()
    n_out = outliers_info[nombre].get(col, 0)
    bp    = ax.boxplot(s, patch_artist=True, vert=True,
                       flierprops=dict(marker="o", markersize=3,
                                       markerfacecolor="red", alpha=0.5))
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.6)
    ax.set_title(f"{info['etiqueta']}\n(outliers IQR×3: {n_out})",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel(info["unidad"], fontsize=8)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G2_boxplots_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G2 guardado: {ruta}")


# ── GRÁFICO 3 — NaN antes y después del tratamiento ──────────────────────────
# Visualiza cuántos valores faltantes había por variable y cómo quedan
# tras el tratamiento. Justifica paso 2.2.E 

# Consolidamos NaN antes y después
nulos_antes_total   = pd.concat(nulos_antes_dict.values())
nulos_despues_total = pd.concat(nulos_despues_dict.values())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("INE — Valores nulos (NaN) antes y después del tratamiento",
             fontsize=13, fontweight="bold")

# Escala común: si no hay NaN usamos 5 como mínimo para que el gráfico
# sea legible y no colapse el eje Y a cero
max_y = max(nulos_antes_total.values) if max(nulos_antes_total.values) > 0 else 5

axes[0].bar(nulos_antes_total.index, nulos_antes_total.values,
            color="#E91E63", alpha=0.8, edgecolor="white")
axes[0].set_title("Antes del tratamiento", fontweight="bold", fontsize=11)
axes[0].set_ylabel("Nº de valores nulos")
axes[0].set_ylim(0, max_y * 1.2)
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(nulos_antes_total.values):
    axes[0].text(i, max_y * 0.05, str(int(v)), ha="center", va="bottom", fontsize=9)

axes[1].bar(nulos_despues_total.index, nulos_despues_total.values,
            color="#4CAF50", alpha=0.8, edgecolor="white")
axes[1].set_title("Después del tratamiento", fontweight="bold", fontsize=11)
axes[1].set_ylabel("Nº de valores nulos")
axes[1].set_ylim(0, max_y * 1.2)  # misma escala que el panel izquierdo
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(nulos_despues_total.values):
    axes[1].text(i, max_y * 0.05, str(int(v)), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G3_nulos_antes_despues_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G3 guardado: {ruta}")


# ── GRÁFICO 4 — Completitud del dataset limpio ────────────────────────────────
# Confirma que el dataset limpio tiene el 100% de datos válidos.
# Justifica la calidad del resultado del paso 2.2.E.

# Completitud dentro del rango propio de cada archivo.
completitud = pd.concat([
    (1 - datasets_clean[n].isna().mean()) * 100
    for n in FILES_INE
])

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(completitud.index, completitud.values,
               color=COLORES[:len(completitud)], alpha=0.8, edgecolor="white")
ax.set_xlim(0, 115)
ax.set_xlabel("% de datos válidos (completitud)", fontsize=11)
ax.set_title("INE — Completitud del dataset limpio por variable\n"
             "(100% = ningún valor nulo dentro del rango propio de cada archivo)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, completitud.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.axvline(100, color="green", linestyle="--", linewidth=1.2,
           alpha=0.7, label="100% completitud")
ax.legend(fontsize=9)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G4_completitud_ine.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G4 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.3.E — RENOMBRADO DE COLUMNAS
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.3.E — RENOMBRADO DE COLUMNAS")
print("="*65)
print(f"\n  {'Archivo':<10} {'Variable resultante':<25} {'Descripción':<40} {'Unidad'}")
print(f"  {'-'*10} {'-'*25} {'-'*40} {'-'*20}")

descripciones = {
    "pib":          "PIB pm índice vol. encadenado Base 2015",
    "tasa_paro":    "Tasa paro = (actividad-empleo)/actividad×100",
    "ipc_var_anual":"IPC variación anual nacional",
}
for nombre, info in VARIABLES_INFO.items():
    col = info["nombre"]
    print(f"  {nombre:<10} {col:<25} {descripciones[col]:<40} {info['unidad']}")

print("\n  Nombres sin acentos ni espacios.")
print("  Variables intermedias (tasa_actividad, tasa_empleo) descartadas.")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2.4.E — EXPLORACIÓN ESTADÍSTICA
# Media, desviación típica, detección de outliers
# Se añade también: mediana, mín, máx, asimetría y curtosis
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 2.4.E — EXPLORACIÓN ESTADÍSTICA")
print("="*65)

for nombre, df in datasets_clean.items():
    col  = VARIABLES_INFO[nombre]["nombre"]
    s    = df[col].dropna()
    print(f"\n  {nombre} — {VARIABLES_INFO[nombre]['etiqueta']}")
    print(f"    Observaciones : {len(s)}")
    print(f"    Rango temporal: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"    Frecuencia    : {VARIABLES_INFO[nombre]['frecuencia']}")
    print(f"    Media         : {s.mean():.4f}  {VARIABLES_INFO[nombre]['unidad']}")
    print(f"    Mediana       : {s.median():.4f}")
    print(f"    Desv. típica  : {s.std():.4f}")
    print(f"    Mínimo        : {s.min():.4f}  (fecha: {s.idxmin().date()})")
    print(f"    Máximo        : {s.max():.4f}  (fecha: {s.idxmax().date()})")
    print(f"    Asimetría     : {s.skew():.4f}  "
          f"({'cola derecha' if s.skew()>0 else 'cola izquierda'})")
    print(f"    Curtosis      : {s.kurt():.4f}  "
          f"({'leptocúrtica' if s.kurt()>3 else 'platicúrtica/mesocúrtica'})")
    print(f"    Coef. variación: {s.std()/s.mean()*100:.2f}%")
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = ((s < Q1 - 3*IQR) | (s > Q3 + 3*IQR)).sum()
    print(f"    Outliers IQR×3: {n_out} "
          f"({'conservados — eventos económicos reales' if n_out > 0 else 'ninguno [OK]'})")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR ARCHIVOS LIMPIOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  GUARDANDO ARCHIVOS LIMPIOS")
print("="*65)

for nombre, df in datasets_clean.items():
    ruta = os.path.join(OUTPUT_PATH, f"{nombre}_limpio.xlsx")
    df.to_excel(ruta)
    print(f"  {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — DATASETS LIMPIOS INE")
print("="*65)

for nombre, df in datasets_clean.items():
    resumen_dataset(nombre, df)

print("\n" + "="*65)
print("  VARIABLES FINALES DEL BLOQUE INE")
print("="*65)
print(f"\n  {'Archivo':<10} {'Variable':<25} {'Unidad':<25} {'Frecuencia'}")
print(f"  {'-'*10} {'-'*25} {'-'*25} {'-'*12}")
for nombre, info in VARIABLES_INFO.items():
    print(f"  {nombre:<10} {info['nombre']:<25} {info['unidad']:<25} {info['frecuencia']}")

print(f"\n   Pipeline INE completado.")
print(f"     Archivos limpios  : {OUTPUT_PATH}")
print(f"     Gráficos generados: {GRAFICOS_PATH}")
print(f"       G1 Histogramas + KDE (distribución variables seleccionadas)")
print(f"       G2 Boxplots (detección visual outliers)")
print(f"       G3 NaN antes vs después del tratamiento")
print(f"       G4 Completitud del dataset limpio")