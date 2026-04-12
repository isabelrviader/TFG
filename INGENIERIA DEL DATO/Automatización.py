# -*- coding: utf-8 -*-
"""
AUTOMATIZACIÓN INGENIERIA DEL DATO

"""

import subprocess, sys

scripts = [
    "EXPLORACIÓN Y LIMPIEZA BdE.py",
    "EXPLORACIÓN Y LIMPIEZA INE.py",
    "EXPLORACIÓN Y LIMPIEZA petróleo.py",
    "TRANSFORMACIÓN BdE.py",
    "TRANSFORMACIÓN INE.py",
    "TRANSFORMACIÓN petróleo.py",
    "UNION.py",
    "EDA.py",
]

ruta_base = r'C:\Users\isabe\Desktop\TFG\CÓDIGO\INGENIERIA DEL DATO'

for script in scripts:
    ruta_completa = f'{ruta_base}\\{script}'
    print(f"\n{'='*60}")
    print(f"Ejecutando: {script}")
    print('='*60)
    resultado = subprocess.run(
        [sys.executable, ruta_completa],
        capture_output=False
    )
    if resultado.returncode == 0:
        print(f"OK: {script} completado")
    else:
        print(f"ERROR: {script} falló con código {resultado.returncode}")
        break