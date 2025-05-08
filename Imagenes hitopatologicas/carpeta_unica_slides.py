import os
import shutil
from pathlib import Path
import argparse
import warnings 


parser = argparse.ArgumentParser(description='Generate features from a given patches')
parser.add_argument('--base_path', default='./', type=str,
                    help='Directorio base donde buscar los archivos .svs')          
parser.add_argument('--target_folder', default='./slides', type=str,
                    help='Nombre de la carpeta destino para los archivos .svs')

args = parser.parse_args()
# Ruta base
base_path = Path(args.base_path)
target_folder = Path(args.target_folder)

# Crear la carpeta si no existe
target_folder.mkdir(parents=True, exist_ok=True)

# Buscar todos los archivos .svs dentro de /content/trident/**/
for svs_file in base_path.rglob("*.svs"):
    # Ignorar los que ya están en la carpeta final
    if svs_file.parent != target_folder:
        dest = target_folder / svs_file.name
        # Si el archivo ya existe en el destino, cambiar el nombre
        if dest.exists():
            print(f"Archivo ya existe: {svs_file.name}, no se moverá.")
        else:
            shutil.move(str(svs_file), str(dest))
            print(f"Archivo movido: {svs_file.name}")
        print(svs_file.parent)

print(f"✅ Imágenes .svs movidas a: {target_folder}")


