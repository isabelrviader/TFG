"""
ANÁLISIS DE NEGOCIO —  WEB CREDITOCLARO
================================================

ACTUALIZACIÓN TRIMESTRAL:
Cuando el BdE e INE publiquen nuevos datos:
1. Actualizar dataset_modelos.xlsx con los nuevos datos
2. Ejecutar COMPARACIÓN.py
3. Ejecutar este script
4. Sube generará index.html con los datos actualizados
"""

import pandas as pd
import numpy as np
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — AJUSTA ESTAS RUTAS A TU ENTORNO
# ──────────────────────────────────────────────────────────────────────────────

RUTA_DATOS  = r'C:\Users\isabe\Desktop\TFG\ENTREGA 2\DATOS\analisis_del_dato\dataset_modelos.xlsx'
RUTA_SALIDA = r'C:\Users\isabe\Desktop\TFG\ANALISIS DE NEGOCIO'
CORREO_CONTACTO = 'creditoclaro@gmail.com'

os.makedirs(RUTA_SALIDA, exist_ok=True)

print("\n" + "="*70)
print("GENERANDO WEB CREDITOCLARO")
print("="*70)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS Y MODELOS
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(RUTA_DATOS)
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').reset_index(drop=True)

FEATURES = ['tasa_paro_lag3', 'credito_hogares_yoy', 'precio_m2_vivienda_yoy_lag4',
            'euribor_12m', 'ipc_var_anual', 'brent_yoy']
TARGET = 'mora_hogares'

mask    = df[FEATURES + [TARGET]].notna().all(axis=1)
X       = df.loc[mask, FEATURES].values
y       = df.loc[mask, TARGET].values   # ya está en M€
fechas  = df.loc[mask, 'fecha']

# Trimestre en formato legible
ultimo_dt = fechas.iloc[-1]
trimestre_num = (ultimo_dt.month - 1) // 3 + 1
meses_trimestre = {1: 'primer', 2: 'segundo', 3: 'tercer', 4: 'cuarto'}
ultimo_trimestre_texto = f"{meses_trimestre[trimestre_num]} trimestre de {ultimo_dt.year}"
ultimo_trimestre_corto = f"{ultimo_dt.year}-Q{trimestre_num}"

print(f"  Último trimestre disponible: {ultimo_trimestre_corto}")

# Entrenar modelos en muestra completa
HIPERPARAMETROS_RF = {
    'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 5,
    'min_samples_split': 10, 'random_state': 42, 'n_jobs': -1
}

rf  = RandomForestRegressor(**HIPERPARAMETROS_RF)
rf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)

valores_actuales = df.loc[mask, FEATURES].iloc[-1].copy()
mora_rf  = rf.predict(valores_actuales.values.reshape(1, -1))[0]
mora_ols = ols.predict(valores_actuales.values.reshape(1, -1))[0]

print(f"  Mora base RF : {mora_rf:.0f} M€")
print(f"  Mora base OLS: {mora_ols:.0f} M€")

# ──────────────────────────────────────────────────────────────────────────────
# PILAR 2 — INDICADORES EN TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────────

indicadores_config = [
    ('tasa_paro_lag3',              'Tasa de paro',                      '%', 'directa', 0.75),
    ('credito_hogares_yoy',         'Crecimiento del crédito a hogares', '%', 'inversa', 0.25),
    ('precio_m2_vivienda_yoy_lag4', 'Precio de la vivienda',             '%', 'inversa', 0.25),
    ('euribor_12m',                 'Euríbor 12M',                       '%', 'directa', 0.75),
    ('ipc_var_anual',               'IPC variación anual',               '%', 'inversa', 0.25),
]

indicadores = []
for col, nombre, unidad, tipo, percentil in indicadores_config:
    umbral = df.loc[mask, col].quantile(percentil)
    valor  = valores_actuales[col]
    alerta = (valor > umbral) if tipo == 'directa' else (valor < umbral)
    indicadores.append({
        'nombre': nombre,
        'valor':  round(float(valor), 2),
        'umbral': round(float(umbral), 2),
        'unidad': unidad,
        'alerta': alerta,
        'tipo':   tipo
    })

alertas_activas = sum(1 for i in indicadores if i['alerta'])

# Euríbor cerca del umbral (para texto especial)
euribor_ind = next(i for i in indicadores if 'Euríbor' in i['nombre'])
euribor_cerca = not euribor_ind['alerta'] and (euribor_ind['umbral'] - euribor_ind['valor']) < 0.5

# Semáforo global
if alertas_activas == 0:
    semaforo        = 'VERDE'
    semaforo_color  = '#16a34a'
    semaforo_bg     = '#f0fdf4'
    semaforo_titulo = 'Sí. El mercado crediticio español está en un momento favorable.'
    semaforo_sub    = 'Es un buen trimestre para plantearse pedir financiación.'
    semaforo_estado = 'Todo tranquilo. Ninguno de los cinco indicadores que analizamos está en zona de riesgo.'
elif alertas_activas <= 2:
    semaforo        = 'AMARILLO'
    semaforo_color  = '#d97706'
    semaforo_bg     = '#fffbeb'
    semaforo_titulo = 'Precaución. Algunos indicadores se acercan a sus niveles de alerta históricos.'
    semaforo_sub    = 'Revisa el análisis completo antes de tomar decisiones importantes.'
    semaforo_estado = f'{alertas_activas} de los cinco indicadores están en zona de atención.'
else:
    semaforo        = 'ROJO'
    semaforo_color  = '#dc2626'
    semaforo_bg     = '#fef2f2'
    semaforo_titulo = 'Riesgo elevado. Varios indicadores superan sus niveles de alerta históricos.'
    semaforo_sub    = 'El ciclo crediticio muestra señales de deterioro. Consulta el análisis completo.'
    semaforo_estado = f'{alertas_activas} de los cinco indicadores están en zona de riesgo.'

print(f"  Semáforo: {semaforo} ({alertas_activas}/5 alertas activas)")

# ──────────────────────────────────────────────────────────────────────────────
# PILAR 3 — ESCENARIOS DE ESTRÉS
# ──────────────────────────────────────────────────────────────────────────────

escenarios_def = [
    {'nombre': 'Base',     'paro': 0,  'euribor': 0,  'credito': 0,   'color': '#f8fafc', 'badge_bg': '#e2e8f0', 'badge_txt': '#475569', 'desc': 'Condiciones actuales'},
    {'nombre': 'Moderado', 'paro': +2, 'euribor': +1, 'credito': -5,  'color': '#fefce8', 'badge_bg': '#fef08a', 'badge_txt': '#854d0e', 'desc': 'Paro +2 pp · Euríbor +1 pp · Crédito -5 pp'},
    {'nombre': 'Severo',   'paro': +4, 'euribor': +2, 'credito': -10, 'color': '#fff7ed', 'badge_bg': '#fed7aa', 'badge_txt': '#9a3412', 'desc': 'Paro +4 pp · Euríbor +2 pp · Crédito -10 pp'},
    {'nombre': 'Extremo',  'paro': +8, 'euribor': +4, 'credito': -20, 'color': '#fef2f2', 'badge_bg': '#fecaca', 'badge_txt': '#991b1b', 'desc': 'Paro +8 pp · Euríbor +4 pp · Crédito -20 pp'},
]

escenarios = []
for esc in escenarios_def:
    vals = valores_actuales.copy()
    vals['tasa_paro_lag3']      += esc['paro']
    vals['euribor_12m']         += esc['euribor']
    vals['credito_hogares_yoy'] += esc['credito']

    m_rf  = rf.predict(vals.values.reshape(1, -1))[0]
    m_ols = ols.predict(vals.values.reshape(1, -1))[0]

    inc_rf  = (m_rf  / mora_rf  - 1) * 100
    inc_ols = (m_ols / mora_ols - 1) * 100

    escenarios.append({**esc,
        'mora_rf':  round(m_rf, 0),
        'mora_ols': round(m_ols, 0),
        'inc_rf':   round(inc_rf, 1),
        'inc_ols':  round(inc_ols, 1),
    })
    print(f"  {esc['nombre']:10s}: RF={m_rf:.0f} M€ ({inc_rf:+.1f}%)  OLS={m_ols:.0f} M€ ({inc_ols:+.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICOS EMBEBIDOS EN BASE64
# ──────────────────────────────────────────────────────────────────────────────

COLOR_AZUL    = '#003366'
COLOR_RF      = '#2563eb'
COLOR_OLS     = '#1e3a5f'

def fig_to_base64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

plt.style.use('seaborn-v0_8-whitegrid')

# ── Gráfico 1: Evolución histórica mora_hogares ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 4.5))
fig1.patch.set_facecolor('white')

mora_serie = df.loc[mask, TARGET].values  # ya en M€

ax1.fill_between(fechas, mora_serie, alpha=0.15, color=COLOR_AZUL)
ax1.plot(fechas, mora_serie, color=COLOR_AZUL, linewidth=2.5)

# Anotación pico — texto desplazado a la izquierda para no tapar el título
idx_max = np.argmax(mora_serie)
pico_val = mora_serie[idx_max]
pico_str = f'{pico_val:,.0f}'.replace(',', '.') + ' M€'
ax1.annotate(f'Pico crisis\n{pico_str}',
             xy=(fechas.iloc[idx_max], pico_val),
             xytext=(-55, -35), textcoords='offset points',
             ha='center', fontsize=8.5, color='#dc2626', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5))

# Anotación último valor — texto desplazado a la izquierda del punto final
ultimo_val = mora_serie[-1]
ultimo_str = f'{ultimo_val:,.0f}'.replace(',', '.') + ' M€'
ax1.annotate(f'{ultimo_trimestre_corto}\n{ultimo_str}',
             xy=(fechas.iloc[-1], ultimo_val),
             xytext=(-55, 20), textcoords='offset points',
             ha='center', fontsize=8.5, color='#16a34a', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5))

# Eje Y con separador de miles con punto
import matplotlib.ticker as mticker
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'.replace(',', '.')))

ax1.set_title('Evolución histórica de la morosidad de hogares en España (2005–2025)',
              fontsize=12, fontweight='bold', pad=12, color=COLOR_AZUL)
ax1.set_ylabel('Mora hogares (M€)', fontsize=9, color='#475569')
ax1.tick_params(colors='#64748b', labelsize=8)
ax1.spines[['top', 'right']].set_visible(False)
fig1.tight_layout()
img_mora = fig_to_base64(fig1)
print("  Grafico 1 (mora historica) generado")

# ── Gráfico 2: Escenarios de estrés ─────────────────────────────────────────
nombres_esc = [e['nombre'] for e in escenarios]
moras_rf    = [e['mora_rf']  for e in escenarios]
moras_ols   = [e['mora_ols'] for e in escenarios]

x_pos = np.arange(len(nombres_esc))
ancho = 0.38

fig2, ax2 = plt.subplots(figsize=(10, 4.5))
fig2.patch.set_facecolor('white')

bars_rf  = ax2.bar(x_pos - ancho/2, moras_rf,  ancho, label='Random Forest', color=COLOR_RF,  alpha=0.85)
bars_ols = ax2.bar(x_pos + ancho/2, moras_ols, ancho, label='OLS (regulatorio)', color=COLOR_OLS, alpha=0.85)

for bar in bars_rf:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
             f'{int(bar.get_height()):,}'.replace(',', '.'),
             ha='center', va='bottom', fontsize=8, color=COLOR_RF, fontweight='bold')
for bar in bars_ols:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
             f'{int(bar.get_height()):,}'.replace(',', '.'),
             ha='center', va='bottom', fontsize=8, color=COLOR_OLS, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(nombres_esc, fontsize=10)
ax2.set_ylabel('Mora estimada (M€)', fontsize=9, color='#475569')

# Eje Y con separador de miles con punto y margen superior para las etiquetas
import matplotlib.ticker as mticker
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{int(x):,}'.replace(',', '.')))
y_max = max(max(moras_rf), max(moras_ols))
ax2.set_ylim(0, y_max * 1.18)

ax2.set_title('Escenarios de estrés macroeconómico\n(estimación RF vs OLS)',
              fontsize=12, fontweight='bold', pad=12, color=COLOR_AZUL)
ax2.legend(fontsize=9)
ax2.tick_params(colors='#64748b', labelsize=8)
ax2.spines[['top', 'right']].set_visible(False)
fig2.tight_layout()
img_estres = fig_to_base64(fig2)
print("  Grafico 2 (escenarios estres) generado")

# ──────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DEL HTML DE INDICADORES
# ──────────────────────────────────────────────────────────────────────────────

descripciones_indicadores = {
    'Tasa de paro': (
        'El indicador más determinante. Cuando el empleo cae, las familias dejan de pagar.',
        'El desempleo no provoca impagos de forma inmediata. Una familia que pierde su trabajo primero agota sus ahorros, luego las prestaciones, y solo entonces deja de pagar la hipoteca. Por eso vigilamos la tasa de paro con tres trimestres de antelación.'
    ),
    'Crecimiento del crédito a hogares': (
        'Cuando el crédito se contrae, el riesgo aumenta.',
        'Cuando los bancos dejan de prestar, el sistema se tensa. Las familias no pueden refinanciar sus deudas, los proyectos se paralizan y la mora sube. Un crédito en crecimiento es señal de que el sistema financiero confía en la economía.'
    ),
    'Euríbor 12M': (
        'El tipo de interés que determina cuánto pagan cada mes millones de hipotecas en España.',
        'La mayoría de las hipotecas en España son a tipo variable. Cuando el Euríbor sube, la cuota mensual sube con él. Recuerda que su peso en nuestro análisis es del 9%, por lo que estar cerca del nivel de alerta no implica un riesgo elevado para el sistema en su conjunto.'
    ),
    'IPC variación anual': (
        'La inflación como señal del estado de la economía.',
        'Una inflación moderada suele ir de la mano de actividad económica y empleo. Cuando los precios caen por debajo de cierto nivel, suele ser señal de que la economía se está enfriando, y con ella la capacidad de pago de los hogares.'
    ),
    'Precio de la vivienda': (
        'El valor del activo que respalda la mayoría de las hipotecas.',
        'Cuando el precio de la vivienda cae, los hogares ven reducida su capacidad de refinanciarse. Si la deuda supera el valor del inmueble, el incentivo para seguir pagando desaparece. Vigilamos este indicador con cuatro trimestres de antelación.'
    ),
}

textos_estado = {
    ('Tasa de paro',                    False): 'Estamos muy lejos del nivel de alerta. Sin riesgo.',
    ('Tasa de paro',                    True):  'Supera el nivel de alerta histórico. Señal de riesgo.',
    ('Crecimiento del crédito a hogares', False): 'El crédito está creciendo. Sin riesgo.',
    ('Crecimiento del crédito a hogares', True):  'El crédito se contrae por debajo del nivel de alerta.',
    ('Euríbor 12M',                     False): 'Cerca del nivel de alerta. Es el indicador que más vigilamos este trimestre.' if euribor_cerca else 'Por debajo del nivel de alerta. Sin riesgo.',
    ('Euríbor 12M',                     True):  'Supera el nivel de alerta histórico.',
    ('IPC variación anual',             False): 'La inflación está en zona saludable. Sin riesgo.',
    ('IPC variación anual',             True):  'La inflación ha caído por debajo del nivel de alerta.',
    ('Precio de la vivienda',           False): 'El precio de la vivienda está subiendo moderadamente. Sin riesgo.',
    ('Precio de la vivienda',           True):  'El precio de la vivienda ha caído por debajo del nivel de alerta.',
}

html_indicadores = ''
for ind in indicadores:
    color_alerta = '#f59e0b' if (not ind['alerta'] and ind['nombre'] == 'Euríbor 12M' and euribor_cerca) else ('#dc2626' if ind['alerta'] else '#16a34a')
    icono_alerta = '⚠' if (not ind['alerta'] and ind['nombre'] == 'Euríbor 12M' and euribor_cerca) else ('✗' if ind['alerta'] else '✓')
    texto_badge  = 'Atención' if (not ind['alerta'] and ind['nombre'] == 'Euríbor 12M' and euribor_cerca) else ('Alerta' if ind['alerta'] else 'Normal')
    bg_card      = '#fffbeb' if (not ind['alerta'] and ind['nombre'] == 'Euríbor 12M' and euribor_cerca) else ('#fff5f5' if ind['alerta'] else '#f0fdf4')
    subtitulo, descripcion = descripciones_indicadores.get(ind['nombre'], ('', ''))
    texto_estado = textos_estado.get((ind['nombre'], ind['alerta']), '')

    html_indicadores += f"""
    <div class="ind-card" style="background:{bg_card}; border-color:{color_alerta}33;">
        <div class="ind-header">
            <div>
                <div class="ind-nombre">{ind['nombre']}</div>
                <div class="ind-subtitulo">{subtitulo}</div>
            </div>
            <span class="ind-badge" style="background:{color_alerta}22; color:{color_alerta};">{icono_alerta} {texto_badge}</span>
        </div>
        <p class="ind-descripcion">{descripcion}</p>
        <div class="ind-valores">
            <div class="ind-val-bloque">
                <div class="ind-val-label">Valor actual</div>
                <div class="ind-val-num" style="color:{color_alerta};">{ind['valor']:+.2f}%</div>
            </div>
            <div class="ind-val-bloque">
                <div class="ind-val-label">Nivel de alerta histórico</div>
                <div class="ind-val-num" style="color:#475569;">{ind['umbral']:+.2f}%</div>
            </div>
        </div>
        <div class="ind-estado" style="color:{color_alerta};">{texto_estado}</div>
    </div>"""

# ──────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DEL HTML DE ESCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

html_escenarios = ''
for esc in escenarios:
    signo_rf  = '+' if esc['inc_rf']  >= 0 else ''
    signo_ols = '+' if esc['inc_ols'] >= 0 else ''
    html_escenarios += f"""
    <div class="sc-card" style="background:{esc['color']}; border-left: 4px solid {esc['badge_bg']};">
        <div class="sc-header">
            <span class="sc-nombre" style="color:{esc['badge_txt']};">{esc['nombre']}</span>
            <span class="sc-desc" style="color:#64748b;">{esc['desc']}</span>
        </div>
        <div class="sc-results">
            <div class="sc-model">
                <div class="sc-model-label">RANDOM FOREST</div>
                <div class="sc-model-val">{int(esc['mora_rf']):,} M€</div>
                <div class="sc-model-inc" style="color:{'#dc2626' if esc['inc_rf'] > 10 else '#64748b'};">{signo_rf}{esc['inc_rf']:.1f}%</div>
            </div>
            <div class="sc-sep">|</div>
            <div class="sc-model">
                <div class="sc-model-label">OLS (REGULATORIO)</div>
                <div class="sc-model-val">{int(esc['mora_ols']):,} M€</div>
                <div class="sc-model-inc" style="color:{'#dc2626' if esc['inc_ols'] > 10 else '#64748b'};">{signo_ols}{esc['inc_ols']:.1f}%</div>
            </div>
        </div>
    </div>"""

# ──────────────────────────────────────────────────────────────────────────────
# HTML COMPLETO
# ──────────────────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CréditoClaro — Sistema de Alerta Temprana Crediticia</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Serif+4:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --azul:       #003366;
            --azul-med:   #1e3a5f;
            --azul-claro: #2563eb;
            --gris-texto: #374151;
            --gris-sub:   #64748b;
            --borde:      #e2e8f0;
            --fondo:      #f8fafc;
            --verde:      #16a34a;
            --amarillo:   #d97706;
            --rojo:       #dc2626;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Source Serif 4', Georgia, serif;
            background: var(--fondo);
            color: var(--gris-texto);
            line-height: 1.7;
        }}

        /* ── HEADER ── */
        .header {{
            background: var(--azul);
            color: white;
            padding: 3rem 2rem 2.5rem;
            text-align: center;
        }}
        .header-logo {{
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 0.6rem;
        }}
        .header-logo span {{ color: #93c5fd; }}
        .header-sub {{
            font-size: 1.05rem;
            color: #bfdbfe;
            max-width: 600px;
            margin: 0 auto 1rem;
            font-weight: 300;
        }}
        .header-update {{
            font-size: 0.82rem;
            color: #93c5fd;
            opacity: 0.8;
        }}

        /* ── SECCIONES ── */
        .section {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2.5rem 1.5rem;
        }}

        .section-title {{
            font-family: 'Playfair Display', serif;
            font-size: 1.7rem;
            color: var(--azul);
            margin-bottom: 0.8rem;
            font-weight: 700;
        }}

        .section-text {{
            font-size: 1rem;
            color: var(--gris-texto);
            margin-bottom: 1rem;
            max-width: 760px;
        }}

        /* ── QUIÉNES SOMOS ── */
        .quienes {{
            background: white;
            border-bottom: 1px solid var(--borde);
        }}
        .quienes .section-text + .section-text {{
            margin-top: 0.8rem;
        }}

        /* ── CONTADOR ── */
        .contador {{
            background: var(--azul-med);
            color: white;
            padding: 2rem 1.5rem;
        }}
        .contador-inner {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .contador-titulo {{
            font-family: 'Playfair Display', serif;
            font-size: 1.05rem;
            color: #93c5fd;
            margin-bottom: 1.5rem;
            text-align: center;
            font-weight: 600;
            max-width: 680px;
            margin-left: auto;
            margin-right: auto;
        }}
        .contador-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.2rem;
            text-align: center;
        }}
        .contador-item {{
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1.4rem 0.8rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .contador-icon {{
            margin-bottom: 0.6rem;
            opacity: 0.9;
        }}
        .contador-num {{
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
            display: block;
        }}
        .contador-label-title {{
            font-size: 0.82rem;
            color: white;
            font-weight: 600;
            margin-top: 0.3rem;
            margin-bottom: 0.4rem;
        }}
        .contador-label-desc {{
            font-size: 0.75rem;
            color: #bfdbfe;
            line-height: 1.5;
        }}

        /* ── SEMÁFORO ── */
        .semaforo-section {{
            background: white;
            border-bottom: 1px solid var(--borde);
        }}
        .semaforo-wrap {{
            display: flex;
            align-items: center;
            gap: 2rem;
            margin: 1.5rem 0;
            background: {semaforo_bg};
            border-radius: 14px;
            padding: 1.8rem 2rem;
            border: 1px solid {semaforo_color}33;
        }}
        .semaforo-circulo {{
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: {semaforo_color};
            flex-shrink: 0;
            box-shadow: 0 0 30px {semaforo_color}55;
        }}
        .semaforo-info h2 {{
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            color: var(--azul);
            margin-bottom: 0.3rem;
        }}
        .semaforo-info p {{
            color: var(--gris-sub);
            font-size: 0.95rem;
        }}
        .semaforo-estado {{
            font-size: 0.9rem;
            color: {semaforo_color};
            font-weight: 600;
            margin-top: 0.4rem;
        }}

        /* ── PERFILES ── */
        .perfiles {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        .perfil-card {{
            background: var(--fondo);
            border: 1px solid var(--borde);
            border-radius: 12px;
            padding: 1.3rem 1.2rem;
            text-align: center;
        }}
        .perfil-icono-svg {{
            display: flex;
            justify-content: center;
            margin-bottom: 0.8rem;
        }}
        .perfil-icono {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
        .perfil-titulo {{
            font-family: 'Playfair Display', serif;
            font-size: 0.95rem;
            color: var(--azul);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .perfil-texto {{ font-size: 0.85rem; color: var(--gris-sub); line-height: 1.6; text-align: left; }}

        /* ── SUSCRIPCIÓN ── */
        .suscripcion {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1.3rem 1.5rem;
            margin-top: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        .suscripcion-texto {{
            font-size: 0.95rem;
            color: var(--azul);
            font-weight: 600;
        }}
        .suscripcion-sub {{
            font-size: 0.82rem;
            color: var(--gris-sub);
            margin-top: 0.2rem;
        }}
        .suscripcion-form {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .suscripcion-input {{
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 0.55rem 1rem;
            font-size: 0.9rem;
            outline: none;
            min-width: 220px;
            font-family: inherit;
        }}
        .suscripcion-btn {{
            background: var(--azul);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.2rem;
            font-size: 0.9rem;
            cursor: pointer;
            font-family: inherit;
            font-weight: 600;
        }}

        /* ── BOTÓN TRANSICIÓN ── */
        .btn-capa2 {{
            display: block;
            text-align: center;
            background: var(--azul);
            color: white;
            padding: 1rem 2rem;
            text-decoration: none;
            font-size: 1rem;
            font-family: 'Source Serif 4', serif;
            transition: background 0.2s;
        }}
        .btn-capa2:hover {{ background: var(--azul-med); }}

        .divider {{ border: none; border-top: 1px solid var(--borde); }}

        /* ── CAPA 2 ── */
        .capa2 {{ background: white; border-bottom: 1px solid var(--borde); }}

        .chart-wrap {{
            background: white;
            border: 1px solid var(--borde);
            border-radius: 12px;
            padding: 1rem;
            margin: 1.2rem 0;
        }}
        .chart-wrap img {{ width: 100%; height: auto; display: block; }}

        /* títulos de subsección */
        .sub-titulo {{
            font-family: 'Playfair Display', serif;
            font-size: 1.35rem;
            color: var(--azul);
            font-weight: 700;
            margin: 2rem 0 0.5rem;
        }}
        .sub-desc {{
            font-size: 0.92rem;
            color: var(--gris-sub);
            margin-bottom: 1.2rem;
        }}

        /* ── INDICADORES ── */
        .indicators-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}
        .ind-card {{
            border: 1px solid;
            border-radius: 12px;
            padding: 1.2rem 1.3rem;
        }}
        .ind-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.6rem;
            gap: 0.5rem;
        }}
        .ind-nombre {{
            font-family: 'Playfair Display', serif;
            font-size: 1.05rem;
            color: var(--azul);
            font-weight: 700;
        }}
        .ind-subtitulo {{
            font-size: 0.8rem;
            color: var(--gris-sub);
            font-style: italic;
            margin-top: 0.1rem;
        }}
        .ind-badge {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            white-space: nowrap;
            flex-shrink: 0;
        }}
        .ind-descripcion {{
            font-size: 0.86rem;
            color: var(--gris-sub);
            margin-bottom: 0.8rem;
            line-height: 1.6;
        }}
        .ind-valores {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 0.6rem;
        }}
        .ind-val-label {{
            font-size: 0.72rem;
            color: var(--gris-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.1rem;
        }}
        .ind-val-num {{
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            font-weight: 700;
        }}
        .ind-estado {{
            font-size: 0.82rem;
            font-weight: 600;
            padding-top: 0.4rem;
            border-top: 1px solid #e2e8f055;
        }}

        /* ── ESCENARIOS ── */
        .scenarios-list {{ display: flex; flex-direction: column; gap: 0.8rem; }}
        .sc-card {{
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
        }}
        .sc-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.8rem;
            flex-wrap: wrap;
        }}
        .sc-nombre {{
            font-family: 'Playfair Display', serif;
            font-size: 1.1rem;
            font-weight: 700;
        }}
        .sc-desc {{ font-size: 0.82rem; }}
        .sc-results {{
            display: flex;
            gap: 2rem;
            align-items: center;
        }}
        .sc-model-label {{
            font-size: 0.7rem;
            color: var(--gris-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .sc-model-val {{
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--azul);
        }}
        .sc-model-inc {{ font-size: 0.85rem; font-weight: 600; }}
        .sc-sep {{ color: var(--borde); font-size: 1.5rem; }}

        /* ── NOTA METODOLÓGICA ── */
        .nota {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            font-size: 0.82rem;
            color: #1e40af;
            margin-top: 1.5rem;
            line-height: 1.6;
        }}

        /* ── CAPA 3 ── */
        .capa3-wrap {{
            padding: 0 1.5rem 2.5rem;
            max-width: 900px;
            margin: 0 auto;
        }}
        .capa3 {{
            background: var(--azul);
            color: white;
            border-radius: 16px;
            padding: 2.5rem 2rem;
            text-align: center;
        }}
        .capa3 h2 {{
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            margin-bottom: 1rem;
        }}
        .capa3 p {{
            font-size: 0.95rem;
            color: #bfdbfe;
            max-width: 600px;
            margin: 0 auto 1.5rem;
            line-height: 1.7;
        }}
        .api-bloque {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1.3rem 1.5rem;
            max-width: 560px;
            margin: 0 auto 1.5rem;
            text-align: left;
        }}
        .api-titulo {{
            font-family: 'Playfair Display', serif;
            font-size: 1.1rem;
            color: white;
            margin-bottom: 0.4rem;
        }}
        .api-desc {{
            font-size: 0.88rem;
            color: #bfdbfe;
            line-height: 1.6;
        }}
        .btn-contacto {{
            display: inline-block;
            background: white;
            color: var(--azul);
            padding: 0.8rem 2rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 700;
            font-size: 0.95rem;
            margin-top: 0.5rem;
            transition: opacity 0.2s;
        }}
        .btn-contacto:hover {{ opacity: 0.9; }}

        /* ── FOOTER ── */
        footer {{
            background: #0f172a;
            color: #94a3b8;
            text-align: center;
            padding: 2rem 1.5rem;
            font-size: 0.82rem;
            line-height: 1.8;
        }}
        footer strong {{ color: #e2e8f0; }}
        footer .disclaimer {{
            margin-top: 1rem;
            font-size: 0.78rem;
            color: #64748b;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }}

        @media (max-width: 650px) {{
            .semaforo-wrap {{ flex-direction: column; text-align: center; }}
            .perfiles {{ grid-template-columns: repeat(2, 1fr); }}
            .indicators-grid {{ grid-template-columns: 1fr; }}
            .contador-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header-logo {{ font-size: 2.2rem; }}
            .sc-results {{ flex-direction: column; gap: 0.8rem; }}
            .sc-sep {{ display: none; }}
        }}
    </style>
</head>
<body>

<!-- ══ HEADER ══ -->
<header class="header">
    <div class="header-logo">Crédito<span>Claro</span></div>
    <p class="header-sub">Análisis del ciclo crediticio español para que tomes decisiones financieras con datos, no con intuición.</p>
    <div class="header-update">Última actualización: {ultimo_trimestre_texto} · Fuentes: Banco de España · INE</div>
</header>

<!-- ══ QUIÉNES SOMOS ══ -->
<div class="quienes">
    <div class="section">
        <p class="section-text"><strong>¿Estás pensando en pedir una hipoteca, hacer crecer tu negocio o comprarte una casa?</strong></p>
        <p class="section-text">Tomar una gran decisión económica sin saber si es el mejor momento es como conducir a ciegas. Tu banco tiene herramientas de sobra para conocer el terreno que pisa; tú, hasta ahora, no las tenías.</p>
        <p class="section-text">CréditoClaro nace para equilibrar la balanza. Queremos ser la brújula para familias que buscan su primer hogar, estudiantes que se independizan, autónomos con proyectos en mente o emprendedores que quieren dar el salto.</p>
        <p class="section-text"><strong>¿Cómo te ayudamos?</strong></p>
        <p class="section-text">Cada tres meses "traducimos" los datos que realmente importan (el Euríbor, el paro, el precio de la vivienda...) y los comparamos con lo que ha pasado en España desde 2005. No te daremos una tabla de números interminable ni un informe aburrido.</p>
        <p class="section-text">El resultado es una señal clara: te decimos si el mercado está en un momento de riesgo bajo, medio o alto. Así de simple.</p>
        <p class="section-text">CréditoClaro es una herramienta gratuita y pensada para todos, tengas o no estudios financieros. Porque la información está ahí fuera, y nuestro trabajo es ponerla en tus manos para que decidas con total confianza.</p>
        <p class="section-text">La información siempre ha estado ahí; nuestro trabajo es que, por fin, sea tuya y juegue a tu favor.</p>
    </div>
</div>

<!-- ══ CONTADOR DE IMPACTO ══ -->
<div class="contador">
    <div class="contador-inner">
        <div class="contador-titulo">Para que puedas dar el siguiente paso con seguridad, hemos analizado a fondo la realidad económica para que tú no tengas que hacerlo:</div>
        <div class="contador-grid">
            <div class="contador-item">
                <span class="contador-num">81</span>
                <div class="contador-label-title">trimestres de experiencia</div>
                <div class="contador-label-desc">Hemos revisado paso a paso todo lo que ha pasado en el mercado desde 2005 para aprender de las lecciones del pasado y que tú no tengas que tropezar con las mismas piedras.</div>
            </div>
            <div class="contador-item">
                <span class="contador-num">5</span>
                <div class="contador-label-title">indicadores clave</div>
                <div class="contador-label-desc">Vigilamos de cerca lo que realmente afecta a tu bolsillo y a tus proyectos (como el Euríbor o el paro), filtrando solo lo que es importante para ti.</div>
            </div>
            <div class="contador-item">
                <span class="contador-num">20</span>
                <div class="contador-label-title">años de historia financiera</div>
                <div class="contador-label-desc">Analizamos las crisis y los éxitos de las últimas dos décadas en España para entender dónde estamos hoy y ofrecerte la mejor guía posible.</div>
            </div>
            <div class="contador-item">
                <span class="contador-num">2</span>
                <div class="contador-label-title">tecnología que habla tu idioma</div>
                <div class="contador-label-desc">Utilizamos modelos avanzados para obtener la máxima precisión, pero te entregamos un resultado claro y sencillo. Porque de nada sirve la información si no te ayuda a decidir con tranquilidad.</div>
            </div>
        </div>
    </div>
</div>

<!-- ══ CAPA 1: SEMÁFORO ══ -->
<div class="semaforo-section">
    <div class="section">
        <h2 class="section-title">¿Es buen momento para dar el paso?</h2>

        <div class="semaforo-wrap">
            <div class="semaforo-circulo"></div>
            <div class="semaforo-info">
                <h2>{semaforo_titulo}</h2>
                <p>{semaforo_sub}</p>
                <div class="semaforo-estado">{semaforo_estado}</div>
            </div>
        </div>

        <h3 style="font-family:'Playfair Display',serif; color:var(--azul); font-size:1.2rem; margin-bottom:0.8rem;">¿Cuál es tu situación?</h3>
        <div class="perfiles">
            <div class="perfil-card">
                <div class="perfil-titulo">Estoy pensando en comprar un piso</div>
                <p class="perfil-texto">El momento es favorable. Los tipos están controlados y el mercado no muestra señales de que vaya a complicarse en los próximos meses. Si llevas tiempo dándole vueltas, ahora no es mal momento para hablar con tu banco.</p>
            </div>
            <div class="perfil-card">
                <div class="perfil-titulo">Quiero pedir un préstamo para mi negocio</div>
                <p class="perfil-texto">Las condiciones macro acompañan. El empleo está estable y el crédito fluye. Eso no significa que sea fácil, pero el entorno no juega en tu contra.</p>
            </div>
            <div class="perfil-card">
                <div class="perfil-titulo">Tengo ahorros y quiero invertir en inmobiliario</div>
                <p class="perfil-texto">El precio de la vivienda sigue subiendo de forma moderada y no hay señales de burbuja en los indicadores que seguimos. El ciclo está en una fase tranquila.</p>
            </div>
            <div class="perfil-card">
                <div class="perfil-titulo">¿Tu caso es diferente?</div>
                <p class="perfil-texto">Si tu situación es única, contáctanos para una asesoría a medida y buscaremos la mejor solución para ti.</p>
            </div>
        </div>

        <div class="suscripcion">
            <div>
                <div class="suscripcion-texto">Recibe el semáforo en tu correo cada trimestre.</div>
                <div class="suscripcion-sub">Te avisamos cuando el sistema actualice sus datos o algún indicador se acerque a su nivel de alerta.</div>
            </div>
            <div class="suscripcion-form">
                <input class="suscripcion-input" type="email" placeholder="tu@email.com">
                <button class="suscripcion-btn">Suscribirme</button>
            </div>
        </div>
    </div>
</div>

<a href="#capa2" class="btn-capa2">Profundiza en el análisis ↓</a>

<!-- ══ CAPA 2: PANEL TÉCNICO ══ -->
<div id="capa2" class="capa2">
    <div class="section">
        <h2 class="section-title">Informe avanzado — Análisis 360º</h2>

        <h3 class="sub-titulo">¿Por qué está en verde? Esto es lo que nos dice el mercado ahora mismo.</h3>
        <p class="section-text">El semáforo no es una opinión. Es el resultado de analizar seis indicadores macroeconómicos con datos reales del Banco de España y el INE, y compararlos con lo que ha ocurrido históricamente en España desde 2005. Para entender dónde estamos hoy, primero hay que ver de dónde venimos.</p>

        <h3 class="sub-titulo">La morosidad en España: 20 años de historia</h3>
        <div class="chart-wrap">
            <img src="data:image/png;base64,{img_mora}" alt="Evolución histórica mora hogares España 2005-2025">
        </div>
        <p class="section-text">En 2013, la morosidad hipotecaria en España alcanzó su máximo histórico: 50.874 M€. Era la consecuencia acumulada de años de desempleo masivo, crédito fácil y precios de vivienda insostenibles. Lo que vino después fue una recuperación lenta pero sostenida. Hoy, en el {ultimo_trimestre_texto}, la mora se sitúa en 17.702 M€, en niveles similares a los de antes de la crisis. Eso no significa que el riesgo haya desaparecido. Significa que ahora mismo el ciclo está en una fase tranquila, y que los indicadores que históricamente han anticipado los problemas no están enviando señales de alarma.</p>

        <h3 class="sub-titulo">¿Qué variables determinan la morosidad?</h3>
        <p class="section-text">No todos los indicadores pesan igual. Nuestro modelo ha analizado 20 años de datos para determinar cuáles han sido históricamente los más determinantes. El resultado es claro: el empleo explica el 61% de lo que ocurre con la mora en España. Le siguen el crédito a hogares con un 16% y el precio de la vivienda con un 13%. El Euríbor explica el 9% en nuestro análisis. El IPC y el precio del petróleo completan el diagnóstico con un peso menor. Tenerlo en cuenta ayuda a interpretar bien las señales: que el Euríbor esté cerca de su nivel de alerta no implica un riesgo elevado para el sistema en su conjunto.</p>

        <h3 class="sub-titulo">Los cinco indicadores que vigilamos</h3>
        <p class="sub-desc">Cada uno tiene un nivel de alerta histórico: el valor a partir del cual, en el pasado, la mora ha subido. Hoy ninguno está en zona de riesgo.</p>
        <div class="indicators-grid">
            {html_indicadores}
        </div>

        <h3 class="sub-titulo">¿Qué tendría que pasar para que el semáforo cambiara?</h3>
        <p class="section-text">El mercado está tranquilo ahora, pero los ciclos económicos siempre se invierten. Por eso simulamos qué pasaría si las condiciones se deterioraran. No para alarmar, sino para que puedas tomar decisiones con los ojos abiertos.</p>
        <p class="section-text">Nuestro sistema combina dos modelos. El primero, Random Forest, aprende de los patrones históricos y ofrece una estimación conservadora. El segundo, OLS, es el modelo que exige la regulación bancaria europea porque permite explicar exactamente por qué sube o baja la mora. Cada escenario muestra la estimación de ambos modelos: la mora se encontraría en algún punto dentro de ese rango.</p>
        <p class="section-text">Cuando hablamos de que el paro sube 2 puntos, significa que aumenta 2 puntos porcentuales sobre su nivel actual. Cuando decimos que el crédito cae, significa que el ritmo de crecimiento del crédito se reduce respecto a hoy. Son hipótesis de trabajo, no predicciones.</p>

        <div class="scenarios-list">
            {html_escenarios}
        </div>

        <div class="chart-wrap" style="margin-top:1.5rem;">
            <img src="data:image/png;base64,{img_estres}" alt="Gráfico escenarios de estrés RF vs OLS">
        </div>

        <p class="section-text" style="margin-top:1rem;">En el escenario más extremo, ambos modelos arrojan estimaciones similares, lo que refuerza la robustez del resultado.</p>

        <div class="nota">
            <strong>Nota metodológica:</strong> El sistema utiliza Random Forest como modelo principal
            (RMSE Test 28,4% inferior al OLS) y OLS como benchmark regulatorio interpretable bajo IFRS 9 y Basilea III.
            La convergencia de ambos modelos en el escenario extremo (+69–71% sobre el nivel base) refuerza la robustez del resultado.
            Dataset: 81 observaciones trimestrales (2005-Q1 a 2025-Q1). Fuentes: Banco de España, INE.
        </div>
    </div>
</div>

<!-- ══ CAPA 3: API / EMPRESAS ══ -->
<div class="capa3-wrap">
    <div class="capa3">
        <h2>¿Eres una empresa y quieres integrar CréditoClaro?</h2>
        <p>Accede a la serie histórica completa desde 2005, con actualización trimestral automática. Integra el semáforo de CréditoClaro en tu plataforma y ofrece a tus usuarios contexto sobre el ciclo crediticio español que va más allá de la comparación de productos.</p>
        <div class="api-bloque">
            <div class="api-titulo">API de datos</div>
            <div class="api-desc">Accede de forma programática a toda la serie histórica desde 2005 con actualización trimestral automática. Integra el semáforo y los indicadores de CréditoClaro directamente en tu plataforma y ofrece a tus usuarios una capa de contexto sobre el ciclo crediticio español. Alineado con IFRS 9 y Basilea III.</div>
        </div>
        <a href="mailto:{CORREO_CONTACTO}" class="btn-contacto">Contactar &rarr; {CORREO_CONTACTO}</a>
    </div>
</div>

<!-- ══ FOOTER ══ -->
<footer>
    <strong>CréditoClaro</strong> es un proyecto académico desarrollado como Trabajo de Fin de Grado en Business Analytics<br>
    en la <strong>Universidad Francisco de Vitoria</strong> &middot; Grado en Business Analytics y ADE &middot; Curso 2025-26<br>
    <strong>Isabel Rodríguez Viader</strong><br><br>
    Datos: <strong>Banco de España</strong> &middot; <strong>INE</strong>
    <div class="disclaimer">
        CréditoClaro analiza el ciclo económico con datos reales, pero no es un asesor financiero. Sus estimaciones se basan en patrones históricos y explican aproximadamente el 40% del comportamiento de la mora, ya que trabaja exclusivamente con datos públicos. Antes de tomar cualquier decisión importante de inversión o financiación, consulta siempre con un profesional.
    </div>
</footer>

<script>
    document.querySelector('.btn-capa2').addEventListener('click', function(e) {{
        e.preventDefault();
        document.getElementById('capa2').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }});
</script>

</body>
</html>"""

# ──────────────────────────────────────────────────────────────────────────────
# GUARDAR HTML
# ──────────────────────────────────────────────────────────────────────────────

ruta_html = os.path.join(RUTA_SALIDA, 'index.html')
with open(ruta_html, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*70}")
print(f"  WEB GENERADA CORRECTAMENTE")
print(f"{'='*70}\n")