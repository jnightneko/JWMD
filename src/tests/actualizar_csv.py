import csv
from pathlib import Path

def generar_fila(nombre, tipo_bebida, defecto1, defecto2, llenado_tipo):
    fila = {
        'filename': nombre,
        'tipo_bebida': tipo_bebida,
        'etiqueta_mal_colocada': defecto1,
        'contenido_incorrecto': defecto2,
        'subllenado': 1 if llenado_tipo == 'subllenado' else 0,
        'llenado': 1 if llenado_tipo == 'llenado' else 0,
        'mediollenado': 1 if llenado_tipo == 'mediollenado' else 0
    }
    return fila

def actualizar_csv(inicio, fin, tipo_bebida, llenado_tipo, etiqueta_defecto, contenido_defecto):
    ruta_csv = Path('../assets') / 'labels.csv'
    
    # Leer datos existentes
    datos = []
    if ruta_csv.exists():
        with open(ruta_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            datos = list(reader)
    
    existentes = {fila['filename'] for fila in datos}

    # Generar nuevos datos
    nuevos = []
    for i in range(inicio, fin + 1):
        nombre = f"img{i:04d}.jpg"
        if nombre in existentes:
            continue  # Evitar duplicados
        fila = generar_fila(nombre, tipo_bebida, etiqueta_defecto, contenido_defecto, llenado_tipo)
        nuevos.append(fila)

    # Unir y ordenar por nombre
    todos = datos + nuevos
    todos.sort(key=lambda x: int(x['filename'][3:6]))

    # Guardar archivo
    campos = ['filename', 'tipo_bebida', 'etiqueta_mal_colocada', 'contenido_incorrecto',
              'subllenado', 'llenado', 'mediollenado']
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todos)
    
    print(f"\n✅ Se agregaron {len(nuevos)} nuevas filas al CSV.\n")

def main():
    print("=== AGREGAR FILAS AL CSV ===")
    
    try:
        inicio = int(input("Número inicial de imagen (ej. 400): "))
        fin = int(input("Número final de imagen (ej. 599): "))
        print("\nTipo de bebida:")
        print("0: Coca Cola\n1: Fanta\n2: Sprite")
        tipo_bebida = int(input("Elige tipo de bebida [0-2]: "))
        
        print("\nTipo de llenado:")
        print("Opciones: subllenado, llenado, mediollenado")
        llenado = input("Tipo de llenado: ").strip().lower()
        
        etiqueta = int(input("\n¿Etiqueta mal colocada? (1 = sí, 0 = no): "))
        contenido = int(input("¿Contenido incorrecto? (1 = sí, 0 = no): "))
        
        actualizar_csv(inicio, fin, tipo_bebida, llenado, etiqueta, contenido)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
