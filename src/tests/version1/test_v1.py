import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize
from modelo import MultiInputModel # Asegúrate de que modelo.py está en la misma carpeta o en el PATH

# --- CONFIGURACIÓN (Asegúrate de que estos valores coincidan EXACTAMENTE con tu entrenamiento) ---
num_tipos_bebida = 3
num_defectos = 2
num_llenado = 3 # Confirma que este fue el valor con el que se entrenó tu modelo

# Ruta al archivo de pesos del modelo
MODEL_WEIGHTS_PATH = "modelo_control_calidad1_weights.pth" # Asegúrate que esta ruta es correcta

# --- DICIONARIOS DE MAPEO (Deben ser EXACTAMENTE los mismos que en el entrenamiento) ---
tipos_bebida_mapping = {'CocaCola': 0, 'Fanta': 1, 'Sprite': 2}
llenado_mapping = {'subllenado': 0, 'llenado_normal': 1, 'mediollenado': 2}

tipos_bebida_mapping_reverse = {v: k for k, v in tipos_bebida_mapping.items()}
llenado_mapping_reverse = {v: k for k, v in llenado_mapping.items()}

nombres_defectos = ['etiqueta_mal_colocada', 'contenido_incorrecto']


# --- INICIALIZACIÓN DEL MODELO Y CARGA DE PESOS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiInputModel(num_tipos_bebida, num_defectos, num_llenado)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()
print(f"Modelo cargado desde {MODEL_WEIGHTS_PATH} y puesto en modo de evaluación.")

# --- INICIALIZACIÓN DE LA CÁMARA ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Intenta con cv2.CAP_DSHOW para Windows

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Asegúrate de que el índice de cámara sea correcto.")
    exit()
print("Cámara abierta correctamente.")


# --- FUNCIÓN DE PREPROCESAMIENTO DE IMAGEN (SIN CAMBIOS AQUÍ) ---
def preprocess_image(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image = resize(image, (224, 224))
    image_np = np.array(image)

    img_color = image_np / 255.0
    img_color_tensor = torch.tensor(img_color, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    img_bordes_np = np.sqrt(sobelx**2 + sobely**2)

    if img_bordes_np.max() > img_bordes_np.min():
        img_bordes_np = (img_bordes_np - img_bordes_np.min()) / (img_bordes_np.max() - img_bordes_np.min())
    else:
        img_bordes_np = np.zeros_like(img_bordes_np)

    img_bordes_np = np.expand_dims(img_bordes_np, axis=-1)
    img_bordes_tensor = torch.tensor(img_bordes_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return img_color_tensor, img_bordes_tensor, img_bordes_np # <<-- MODIFICACIÓN: devolver también img_bordes_np para visualización


# --- BUCLE PRINCIPAL DE INFERENCIA EN TIEMPO REAL (MODIFICADO PARA VISUALIZACIÓN) ---
print("\nIniciando inferencia en tiempo real. Presiona 'q' para salir.")
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame. Terminando el programa.")
            break

        # Preprocesar el frame (ahora devuelve también img_bordes_np)
        img_color, img_bordes, img_bordes_visualizable = preprocess_image(frame) # <<-- MODIFICACIÓN

        # Mover los tensores de entrada al dispositivo del modelo
        img_color = img_color.to(device)
        img_bordes = img_bordes.to(device)

        # Realizar la predicción
        out_tipo, out_defectos, out_llenado = model(img_color, img_bordes)

        # Interpretar resultados (logits a probabilidades/clases)
        tipo_pred_idx = torch.argmax(torch.softmax(out_tipo, dim=1), dim=1).item()
        tipo_pred_nombre = tipos_bebida_mapping_reverse[tipo_pred_idx]

        llenado_pred_idx = torch.argmax(torch.softmax(out_llenado, dim=1), dim=1).item()
        llenado_pred_nombre = llenado_mapping_reverse[llenado_pred_idx]

        defectos_probs = torch.sigmoid(out_defectos).squeeze().cpu().numpy()
        defectos_detectados = [nombres_defectos[i] for i, prob in enumerate(defectos_probs) if prob > 0.5]
        defectos_str = ", ".join(defectos_detectados) if defectos_detectados else "Ninguno"

        # --- Mostrar resultados y frames ---
        # Mostrar el frame original de la cámara (con resultados)
        cv2.putText(frame, f"Tipo: {tipo_pred_nombre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Llenado: {llenado_pred_nombre}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Defectos: {defectos_str}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Original + Predicciones', frame)
        
        bordes_display = (img_bordes_visualizable * 255).astype(np.uint8)
        # Convertir a 3 canales para que cv2.imshow lo muestre en color (si quieres un visualización RGB, si no, es gris)
        bordes_display_rgb = cv2.cvtColor(bordes_display, cv2.COLOR_GRAY2BGR)

        cv2.imshow('Filtro de Bordes (Sobel)', bordes_display_rgb) # Muestra la imagen con el filtro Sobel
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FIN NUEVO CÓDIGO >>>>>>>>>>>>>>>>>>>>>>>>>>


        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")