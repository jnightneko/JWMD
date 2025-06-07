import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize
from modelo import MultiInputModel # Asegúrate de que modelo.py está en la misma carpeta o en el PATH

# --- CONFIGURACIÓN (Asegúrate de que estos valores coincidan EXACTAMENTE con tu entrenamiento) ---
# Si num_tipos_bebida en entrenamiento fue 3, aquí 3.
num_tipos_bebida = 3
# Si num_defectos en entrenamiento fue 2, aquí 2.
num_defectos = 2

# Si tu modelo fue entrenado con 3 clases de llenado: 'subllenado', 'llenado_normal', 'mediollenado'
# Y sus índices fueron 0, 1, 2 respectivamente.
num_llenado = 3 # Confirma que este fue el valor con el que se entrenó tu modelo

# Ruta al archivo de pesos del modelo
MODEL_WEIGHTS_PATH = "modelo_control_calidad1_weights.pth" # Asegúrate que esta ruta es correcta

# --- DICIONARIOS DE MAPEO (Deben ser EXACTAMENTE los mismos que en el entrenamiento) ---
# Revisa tu script ia.py en la sección de mapeos (Bloque 3)
tipos_bebida_mapping = {'CocaCola': 0, 'Fanta': 1, 'Sprite': 2} # Orden alfabético o el que sea según df['tipo_bebida'].unique()
# Mapeo de llenado: crucial que el orden y los valores coincidan con el entrenamiento
# Si "llenado" de tu CSV se mapeó a la clase 1 y "mediollenado" a la clase 2.
llenado_mapping_reverse = {0: 'subllenado', 1: 'llenado', 2: 'mediollenado'}

tipos_bebida_mapping_reverse = {v: k for k, v in tipos_bebida_mapping.items()}


nombres_defectos = ['etiqueta_mal_colocada', 'contenido_incorrecto']


# --- INICIALIZACIÓN DEL MODELO Y CARGA DE PESOS ---
# Asegúrate de que el modelo esté en el dispositivo correcto (CPU para este caso, o 'cuda' si tienes GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiInputModel(num_tipos_bebida, num_defectos, num_llenado)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device)) # map_location para asegurar que carga en la CPU si entrenaste en GPU
model.eval() # Poner el modelo en modo de evaluación para inferencia (desactiva dropout, etc.)
print(f"Modelo cargado desde {MODEL_WEIGHTS_PATH} y puesto en modo de evaluación.")

# --- INICIALIZACIÓN DE LA CÁMARA ---
# Usa el índice de cámara 1 con el backend DSHOW para Windows, si la cámara por defecto (0) falla.
# Si 1 no funciona, prueba con 0, 2, 3...
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Intenta con cv2.CAP_DSHOW para Windows

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Asegúrate de que el índice de cámara sea correcto.")
    exit()
print("Cámara abierta correctamente.")


# --- FUNCIÓN DE PREPROCESAMIENTO DE IMAGEN (CORREGIDA) ---
def preprocess_image(frame):
    # Convertir de BGR (OpenCV) a RGB (PIL/NumPy)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image) # Convertir a PIL Image para resize de torchvision

    # Redimensionar la imagen a 224x224 (como en el entrenamiento)
    image = resize(image, (224, 224))
    image_np = np.array(image) # Convertir a NumPy array para procesamiento adicional

    # Rama de color:
    # Normalización a 0-1 y permutación a C, H, W. Luego añadir dimensión de batch.
    img_color = image_np / 255.0
    img_color_tensor = torch.tensor(img_color, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # [H, W, C] -> [C, H, W] -> [1, C, H, W]

    # Rama de bordes: (usando Sobel como en el entrenamiento)
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    img_bordes_np = np.sqrt(sobelx**2 + sobely**2)

    # Normalización del tensor de bordes a 0-1
    if img_bordes_np.max() > img_bordes_np.min():
        img_bordes_np = (img_bordes_np - img_bordes_np.min()) / (img_bordes_np.max() - img_bordes_np.min())
    else:
        img_bordes_np = np.zeros_like(img_bordes_np)

    # Añadir dimensión de canal al final y luego permutar (como en el entrenamiento)
    # [H, W] -> [H, W, 1] -> [1, H, W] -> [1, 1, H, W]
    img_bordes_np = np.expand_dims(img_bordes_np, axis=-1) # Añadir dimensión de canal al final
    img_bordes_tensor = torch.tensor(img_bordes_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # Permutar y añadir dimensión de batch

    return img_color_tensor, img_bordes_tensor


# --- BUCLE PRINCIPAL DE INFERENCIA EN TIEMPO REAL ---
print("\nIniciando inferencia en tiempo real. Presiona 'q' para salir.")
with torch.no_grad(): # Desactivar el cálculo de gradientes durante la inferencia
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame. Terminando el programa.")
            break

        # Preprocesar el frame
        img_color, img_bordes = preprocess_image(frame)

        # Mover los tensores de entrada al dispositivo del modelo
        img_color = img_color.to(device)
        img_bordes = img_bordes.to(device)

        # Realizar la predicción
        out_tipo, out_defectos, out_llenado = model(img_color, img_bordes)

        # Interpretar resultados (logits a probabilidades/clases)
        # Tipo de Bebida (Clasificación multi-clase, usa softmax)
        tipo_pred_idx = torch.argmax(torch.softmax(out_tipo, dim=1), dim=1).item()
        tipo_pred_nombre = tipos_bebida_mapping_reverse[tipo_pred_idx] # Asegúrate de tener este diccionario definido

        # Nivel de Llenado (Clasificación multi-clase, usa softmax)
        llenado_pred_idx = torch.argmax(torch.softmax(out_llenado, dim=1), dim=1).item()
        llenado_pred_nombre = llenado_mapping_reverse[llenado_pred_idx]

        # Defectos (Clasificación multi-etiqueta, usa sigmoid y un umbral)
        defectos_probs = torch.sigmoid(out_defectos).squeeze().cpu().numpy() # Mover a CPU antes de numpy()
        defectos_detectados = [nombres_defectos[i] for i, prob in enumerate(defectos_probs) if prob > 0.5]
        # Mostrar "Ninguno" si no hay defectos
        defectos_str = ", ".join(defectos_detectados) if defectos_detectados else "Ninguno"


        # --- Mostrar resultados en la ventana de OpenCV ---
        # Mostrar el tipo de bebida
        cv2.putText(frame, f"Tipo: {tipo_pred_nombre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Mostrar el nivel de llenado
        cv2.putText(frame, f"Llenado: {llenado_pred_nombre}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # Mostrar los defectos
        cv2.putText(frame, f"Defectos: {defectos_str}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar el frame procesado
        cv2.imshow('Control de Calidad en Tiempo Real', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")