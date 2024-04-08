"""
Author: ravargas.42t@gmail.com
Config.py (c) 2024
Desc: description
Created:  2024-04-08T10:13:42.445Z
Modified: !date!
"""
import sys
import os

# Obtener la ruta del directorio del proyecto
pathname = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# Agregar la ruta del proyecto al sys.path si no está ya incluida
if pathname not in sys.path:
    sys.path.append(pathname)

this_dir = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(this_dir, '..', '..', '..', '..', '..'))
FOLDERS = os.listdir(ROOT_DIR)
#specific dirs
CHARTS_DIR = os.path.join(ROOT_DIR, "Charts")

def getDataPaths(round: int):
    data_files = os.listdir(os.path.join(ROOT_DIR, "Data", f"round_{round}"))
    data_files = [os.path.join(ROOT_DIR, "Data", f"round_{round}", i) for i in data_files if not ".zip" in i]
    return data_files