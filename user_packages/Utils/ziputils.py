"""
Author: ravargas.42t@gmail.com
ziputils.py (c) 2024
Desc: contains a function that extracts from zipfile
Created:  2024-04-08T09:31:07.249Z
Modified: !date!
"""


import zipfile as z
import os

def unzip(round: int):

    this_dir = os.getcwd()
    data_dir = os.path.join(this_dir, '..', '..', "Data", f"round_{round}")
    data_files = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]

    if data_files[0].split(".")[-1] != 'zip':
        for files in data_files:
            if files.split(".")[-1] != 'zip':
                os.remove(files)
        return 
    else:
        for files in data_files:
            with z.ZipFile(files, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    # Verificar si el nombre del archivo no comienza con '__MACOSX'
                    if not member.filename.startswith('__MACOSX'):
                        zip_ref.extract(member, data_dir)