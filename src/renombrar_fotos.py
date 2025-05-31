import os
import sys
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

def obtener_fecha_exif(imagen_path):
    try:
        with Image.open(imagen_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"⚠️ No se pudo leer EXIF de {imagen_path.name}: {e}")
    return None  # Si no tiene EXIF o no se puede leer

def renombrar_fotos(directorio, inicio):
    directorio = Path(directorio)
    archivos = list(directorio.glob('*'))
    imagenes = [f for f in archivos if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    # Obtener fecha EXIF si se puede, si no, usar fecha de modificación
    imagenes_con_fecha = []
    for img in imagenes:
        fecha = obtener_fecha_exif(img)
        if not fecha:
            fecha = datetime.fromtimestamp(img.stat().st_mtime)
        imagenes_con_fecha.append((img, fecha))

    # Ordenar por fecha
    imagenes_con_fecha.sort(key=lambda x: x[1])

    numero = inicio
    for img, fecha in imagenes_con_fecha:
        nueva_ext = img.suffix.lower()
        nuevo_nombre = f"img{numero:04d}{nueva_ext}"
        nueva_ruta = directorio / nuevo_nombre

        if nueva_ruta.exists():
            print(f"⚠️ El archivo {nuevo_nombre} ya existe. Saltando.")
        else:
            img.rename(nueva_ruta)
            print(f"✅ {img.name} → {nuevo_nombre}")
            numero += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python renombrar_fotos.py [inicio] [ruta_opcional]")
        print("Ejemplo: python renombrar_fotos.py 1 ./mis_fotos")
        sys.exit(1)

    inicio = int(sys.argv[1])
    ruta = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    renombrar_fotos(ruta, inicio)
