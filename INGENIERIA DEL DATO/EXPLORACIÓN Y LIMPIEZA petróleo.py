# -*- coding: utf-8 -*-
"""
EXPLORACIÓN Y LIMPIEZA — PETRÓLEO (Brent Crude Oil) 

Fuente:                                                             
· CMOHistoricalDataMonthly.xlsx                                     
  World Bank Commodity Markets

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

FILE_PETROLEO = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\CMOHistoricalDataMonthly.xlsx'
OUTPUT_PATH   = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\limpios'
GRAFICOS_PATH = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\GRAFICOS\PETROLEO'

os.makedirs(OUTPUT_PATH,   exist_ok=True)
os.makedirs(GRAFICOS_PATH, exist_ok=True)

# Estilo visual 
plt.style.use("seaborn-v0_8-whitegrid")
COLOR_BRENT = "#FF9800"


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3.1.E — EXPLORACIÓN INICIAL Y CAMBIO DE FORMATO FECHA
#
#   · Filas 0-3: metadatos (título, descripción, fecha actualización)
#   · Fila 4   : nombres de variables (Crude oil Brent, WTI, Dubai...)
#   · Fila 5   : unidades ($/bbl)
#   · Fila 6+  : datos. Columna 0 = fecha 'YYYYMNN' (ej: '1960M01')
#
# Se selecciona la columna 2 (Crude oil, Brent) porque es la referencia
# europea del precio del petróleo, la más relevante para España.
#
# Las fechas se convierten a DatetimeIndex mensual con pd.to_datetime.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  PASO 3.1.E — EXPLORACIÓN INICIAL Y CAMBIO DE FORMATO FECHA")
print("="*65)

# Carga del archivo raw completo para mostrar estado inicial
df_raw = pd.read_excel(FILE_PETROLEO, sheet_name='Monthly Prices', header=None)

print(f"\n  Archivo raw:")
print(f"    Shape original        : {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
print(f"    Filas 0-3             : metadatos (título, descripción)")
print(f"    Fila 4                : nombres de variables")
print(f"    Fila 5                : unidades")
print(f"    Fila 6+               : datos con fechas en col 0")
print(f"\n  Primeras 7 filas (metadatos + cabecera):")
print(df_raw.iloc[:7, :3].to_string())

# Extraer datos: columna 0 (fecha) y columna 2 (Brent)
df = df_raw.iloc[6:, [0, 2]].copy()
df.columns = ['fecha', 'brent']
df = df[df['fecha'].notna()].reset_index(drop=True)

# ── Resumen inicial del archivo raw ──────────────────────────────────────────
print(f"\n  Estado inicial del archivo (raw original):")
print(f"    Filas totales         : {df_raw.shape[0]}")
print(f"    Columnas totales      : {df_raw.shape[1]} (variables de commodities)")
print(f"    Filas de metadatos    : 6 (filas 0-5)")
print(f"    Filas de datos        : {df_raw.shape[0] - 6}")
print(f"    NaN en archivo raw    : {df_raw.isna().sum().sum()}")
print(f"    Tipos de dato raw     : {dict(df_raw.dtypes.value_counts())}")
print(f"    Periodo disponible    : 1960M01 → 2025M12")
print(f"    Duplicados en cols    : {df_raw.columns.duplicated().sum()}")

print(f"\n  Variable seleccionada: Crude oil, Brent (col 2) — referencia europea ✔")
print(f"  Descartadas: crude_avg, Dubai, WTI (menos relevantes para España)")

# Conversión de fechas: '2000M01' → datetime
def parse_fecha_petroleo(s):
    """Convierte '2000M01' → datetime 2000-01-01"""
    try:
        anio, mes = str(s).strip().split('M')
        return pd.to_datetime(f"{anio}-{mes}-01")
    except Exception:
        return pd.NaT

df['fecha'] = df['fecha'].apply(parse_fecha_petroleo)
df = df.dropna(subset=['fecha']).set_index('fecha').sort_index()
df['brent'] = pd.to_numeric(df['brent'], errors='coerce')

print(f"\n  Formato de fecha convertido: 'YYYYMNN' → DatetimeIndex mensual")
print(f"\n  Rango temporal completo: {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Observaciones totales  : {df.shape[0]}")
print(f"  Frecuencia             : mensual")
print(f"  Duplicados en índice   : {df.index.duplicated().sum()}")
print(f"\n  df.head(3):")
print(df.head(3).to_string())
print(f"\n  df.tail(3):")
print(df.tail(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3.2.E — SELECCIÓN DE VARIABLE, RENOMBRADO,
#              TRATAMIENTO DE NULOS Y OUTLIERS
#
# · Variable: brent (ya seleccionada en 3.1.E)
# · Nulos: ninguno en la serie Brent desde 1960
# · Outliers: método IQR×3. Umbral conservador porque los precios
#   extremos del petróleo son eventos geopolíticos reales
#   (embargo árabe 1973, guerra del Golfo, COVID 2020...)
#   Se conservan todos.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 3.2.E — SELECCIÓN, RENOMBRADO, NULOS Y OUTLIERS")
print("="*65)

# Diagnóstico de nulos
n_nulos = df['brent'].isna().sum()
print(f"\n  Nulos en brent: {n_nulos}" if n_nulos == 0
      else f"\n   Nulos en brent: {n_nulos}")

if n_nulos > 0:
    print("  → Aplicando interpolación temporal...")
    df['brent'] = df['brent'].interpolate(method='time', limit_direction='both')
    print(f"  Nulos tras interpolación: {df['brent'].isna().sum()}")

# Detección de outliers IQR×3
s  = df['brent'].dropna()
Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
IQR     = Q3 - Q1
mask    = (s < Q1 - 3*IQR) | (s > Q3 + 3*IQR)
n_out   = mask.sum()

print(f"\n  Detección outliers (IQR × 3):")
print(f"    Q1             : {Q1:.2f} $/bbl")
print(f"    Q3             : {Q3:.2f} $/bbl")
print(f"    IQR            : {IQR:.2f}")
print(f"    Límite inferior: {round(Q1-3*IQR,2):.2f} $/bbl")
print(f"    Límite superior: {round(Q3+3*IQR,2):.2f} $/bbl")
print(f"    Outliers IQR×3 : {n_out} "
      + ("ninguno" if n_out == 0
         else "→ se conservan (eventos geopolíticos reales)"))

if n_out > 0:
    print(f"    Valores extremos detectados:")
    print(df['brent'][mask].to_string())

# El nombre 'brent' ya es semántico y en snake_case — no requiere renombrado
print(f"\n  Nombre de variable: 'brent' ")
print(f"  Unidad            : $/bbl (dólares por barril)")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3.3.E — EXPLORACIÓN ESTADÍSTICA
#
# Estadísticos completos: media, mediana, desv. típica, mín, máx,
# asimetría, curtosis, coeficiente de variación y outliers IQR×3.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("  PASO 3.3.E — EXPLORACIÓN ESTADÍSTICA")
print("="*65)

s = df['brent'].dropna()
print(f"\n  Variable          : brent (Crude oil, Brent)")
print(f"  Observaciones     : {len(s)}")
print(f"  Rango temporal    : {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Frecuencia        : mensual")
print(f"  Media             : {s.mean():.2f} $/bbl")
print(f"  Mediana           : {s.median():.2f} $/bbl")
print(f"  Desv. típica      : {s.std():.2f} $/bbl")
print(f"  Mínimo            : {s.min():.2f} $/bbl  (fecha: {s.idxmin().date()})")
print(f"  Máximo            : {s.max():.2f} $/bbl  (fecha: {s.idxmax().date()})")
print(f"  Asimetría         : {s.skew():.4f}  "
      f"({'cola derecha' if s.skew() > 0 else 'cola izquierda'})")
print(f"  Curtosis          : {s.kurt():.4f}  "
      f"({'leptocúrtica' if s.kurt() > 3 else 'platicúrtica/mesocúrtica'})")
print(f"  Coef. variación   : {s.std()/s.mean()*100:.2f}%")
print(f"  Outliers IQR×3    : {n_out} "
      + ("Ninguno" if n_out == 0 else "→ conservados"))

print(f"\n  df.describe():")
print(df.describe().round(2).to_string())


# ── GRÁFICO G1 — Serie temporal Brent completa ───────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df.index, df['brent'], color=COLOR_BRENT, linewidth=1.2, alpha=0.9)
ax.fill_between(df.index, df['brent'], alpha=0.1, color=COLOR_BRENT)
ax.set_title("Petróleo Brent — Serie temporal mensual completa (1960-2025)\n"
             "(precio en dólares por barril, $/bbl)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("$/bbl")
ax.tick_params(axis="x", rotation=30)

# Eventos clave
eventos = {
    '1973-10-01': ('Embargo árabe', 'top'),
    '1980-01-01': ('Guerra Iran-Iraq', 'top'),
    '2008-07-01': ('Máximo histórico', 'top'),
    '2020-04-01': ('COVID-19', 'bottom'),
    '2022-06-01': ('Guerra Ucrania', 'top'),
}
for fecha, (texto, pos) in eventos.items():
    try:
        x = pd.to_datetime(fecha)
        y = df.loc[x:x]['brent'].values[0] if x in df.index else df['brent'].asof(x)
        vert = 20 if pos == 'top' else -25
        ax.annotate(texto, xy=(x, y),
                    xytext=(0, vert), textcoords='offset points',
                    fontsize=7, ha='center', color='#555555',
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))
    except Exception:
        pass

plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G1_serie_temporal_brent.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  G1 guardado: {ruta}")


# ── GRÁFICO G2 — Histograma + KDE ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(s, bins=40, color=COLOR_BRENT, alpha=0.55, edgecolor="white",
        density=True, label="Histograma")
s.plot.kde(ax=ax, color=COLOR_BRENT, linewidth=2, label="KDE")
ax.axvline(s.mean(),   color="black", linestyle="--", linewidth=1.2,
           label=f"Media: {s.mean():.1f} $/bbl")
ax.axvline(s.median(), color="gray",  linestyle=":",  linewidth=1.2,
           label=f"Mediana: {s.median():.1f} $/bbl")
ax.set_title("Petróleo Brent — Histograma + KDE\n"
             "(distribución del precio mensual 1960-2025)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("$/bbl")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.legend(fontsize=9)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G2_histograma_brent.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G2 guardado: {ruta}")


# ── GRÁFICO G3 — Boxplot (detección visual outliers) ─────────────────────────
fig, ax = plt.subplots(figsize=(6, 7))
bp = ax.boxplot(s, patch_artist=True, vert=True,
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor="red", alpha=0.5))
bp["boxes"][0].set_facecolor(COLOR_BRENT)
bp["boxes"][0].set_alpha(0.6)
ax.set_title(f"Petróleo Brent — Boxplot\n(outliers IQR×3: {n_out})",
             fontsize=12, fontweight="bold")
ax.set_ylabel("$/bbl")
ax.set_xticks([])
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G3_boxplot_brent.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G3 guardado: {ruta}")


# ── GRÁFICO G4 — Completitud del dataset limpio ───────────────────────────────
completitud = (1 - df.isna().mean()) * 100

fig, ax = plt.subplots(figsize=(7, 3))
bar = ax.barh(['brent'], completitud.values,
              color=COLOR_BRENT, alpha=0.8, edgecolor="white")
ax.set_xlim(0, 115)
ax.set_xlabel("% de datos válidos (completitud)", fontsize=11)
ax.set_title("Petróleo — Completitud del dataset limpio\n"
             "(100% = ningún valor nulo)",
             fontsize=12, fontweight="bold")
ax.text(completitud.values[0] + 0.5, 0,
        f"{completitud.values[0]:.1f}%", va="center",
        fontsize=11, fontweight="bold")
ax.axvline(100, color="green", linestyle="--", linewidth=1.2,
           alpha=0.7, label="100% completitud")
ax.legend(fontsize=9)
plt.tight_layout()
ruta = os.path.join(GRAFICOS_PATH, "G4_completitud_brent.png")
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.show()
print(f"  G4 guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR ARCHIVO LIMPIO
# ══════════════════════════════════════════════════════════════════════════════

ruta_output = os.path.join(OUTPUT_PATH, "petroleo_limpio.xlsx")
df.to_excel(ruta_output)
print(f"\n  petroleo_limpio.xlsx guardado: {ruta_output}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RESUMEN FINAL — EXPLORACIÓN Y LIMPIEZA PETRÓLEO")
print("="*65)
print(f"\n  {'Variable':<15} {'Unidad':<12} {'Fuente':<25} {'Frecuencia'}")
print(f"  {'-'*15} {'-'*12} {'-'*25} {'-'*10}")
print(f"  {'brent':<15} {'$/bbl':<12} {'World Bank Pink Sheet':<25} {'Mensual'}")
print(f"\n  Observaciones : {df.shape[0]}")
print(f"  Rango temporal: {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Nulos         : {df['brent'].isna().sum()} ")
print(f"  Outliers IQR×3: {n_out} "
      + ("Ninguno" if n_out == 0 else "→ conservados"))
print(f"\n  Exploración y limpieza petróleo completada.")
print(f"     Output  : {ruta_output}")
print(f"     Gráficos:")
print(f"       G1  Serie temporal Brent completa (1960-2025)")
print(f"       G2  Histograma + KDE")
print(f"       G3  Boxplot (detección visual outliers)")
print(f"       G4  Completitud del dataset limpio")