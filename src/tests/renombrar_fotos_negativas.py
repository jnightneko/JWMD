import pandas as pd
from pathlib import Path

ruta = Path('../assets') / 'labels.csv'
df = pd.read_csv(ruta)


new_rows = []
for i in range(1, 501):
    filename = f"neg_{i:05d}.jpg"
    new_row = {
        "filename": filename,
        "tipo_bebida": 3,
        "etiqueta_mal_colocada": 0,
        "contenido_incorrecto": 0,
        "subllenado": 0,
        "llenado": 0,
        "mediollenado": 0
    }
    new_rows.append(new_row)


new_df = pd.DataFrame(new_rows)


df = pd.concat([df, new_df], ignore_index=True)


df.to_csv('labels.csv', index=False)

print("Se han agregado 500 nuevas filas.")
