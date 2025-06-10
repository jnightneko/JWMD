import os
import sys
import serial
import requests
import cv2
import torch
import numpy as np
import datetime
import time

from PIL import Image
from torchvision.transforms.functional import resize
from modelo import MultiInputModel
from ultralytics import YOLO
from dto import Botella, toJSONBotella
from dotenv import load_dotenv

ESTADO_BUENO = "0"
ESTADO_MALO = "1"
TIEMPO_ESPERA = 3 #segundos

"""
    Función encargada de abrir una conexión serie con arduino, pedendiedo el sistema operativo
    empleada, para enviar datos se emplea de la siguiente manera:

    1. Abrir una conexión
        conn = serialConnect()

    2. Enviar mensajes
        sendSerialData(ESTADO_BUENO, conn)
"""
def serialConnect():
    load_dotenv()

    port = os.getenv('SERIAL_PORT')
    arduino = serial.Serial(port, 9600)
    return arduino

def sendSerialData(msg: str, serial) -> None:
    serial.write(msg.encode())

"""
    Para enviar un registro a la base de datos, se puede utilizar esta función de
    la siguiente manera:

    1. Genere un objeto de tipo botella
        botella = Botella()
        botella.descripcion = 'Botella de prueba'
        botella.estado = 1
        botella.fecha = '04/05/2025'
        botella.imagen = 'img.jpg'
        botella.ruta = '/home/dev'

    2. Envia los datos
        sendAPIData(botella)
"""
def sendAPIData(data: Botella):
    load_dotenv()
    URL_SERVER = 'http://' + os.getenv('SERVER_NAME') + ':' + os.getenv('SERVER_PORT') + '/botella'
    requests.post(URL_SERVER, json=toJSONBotella(data))

# --- FUNCIONES ---
def preprocess_image(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil_rgb = resize(Image.fromarray(image_rgb), (224, 224))
    image_np_rgb = np.array(image_pil_rgb) / 255.0
    img_color_tensor = torch.tensor(image_np_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(image_gray, threshold1=100, threshold2=200)
    image_pil_edges = resize(Image.fromarray(image_edges), (224, 224))
    image_np_edges = np.array(image_pil_edges) / 255.0
    img_bordes_tensor = torch.tensor(image_np_edges, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return img_color_tensor, img_bordes_tensor


def hay_botella_con_yolo(frame):
    results = yolo_model(frame, verbose=False)[0]
    for r in results.boxes:
        cls_id = int(r.cls)
        cls_name = yolo_model.names[cls_id]
        if "bottle" in cls_name.lower():
            return True
    return False


def main(args) -> int:
    # --- CONEXIÓN CON ARDUINO ---
    arduino = serialConnect()

    # --- CONFIGURACIÓN ---
    num_tipos_bebida = 3  # Modificado para que coincida con el modelo
    num_defectos = 2
    num_llenado = 3  # Modificado para que coincida con el modelo
    MODEL_WEIGHTS_PATH = os.getenv('MODEL_WEIGHTS_PATH')

    # Mapeos compatibles con el modelo entrenado (3 clases)
    tipos_bebida_mapping = {"CocaCola": 0, "Fanta": 1, "Sprite": 2}
    llenado_mapping = {"subllenado": 0, "llenado_normal": 1, "mediollenado": 2}
    tipos_bebida_mapping_reverse = {v: k for k, v in tipos_bebida_mapping.items()}
    llenado_mapping_reverse = {v: k for k, v in llenado_mapping.items()}
    nombres_defectos = ["etiqueta_mal_colocada", "contenido_incorrecto"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiInputModel(num_tipos_bebida, num_defectos, num_llenado)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo cargado desde {MODEL_WEIGHTS_PATH} y puesto en modo de evaluación.")

    yolo_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()
    print("Cámara abierta correctamente.")


    # --- LOOP PRINCIPAL ---
    print("\nIniciando inferencia. Presiona 'q' para salir.")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error leyendo frame.")
                break

            if hay_botella_con_yolo(frame):
                img_color, img_bordes = preprocess_image(frame)
                img_color = img_color.to(device)
                img_bordes = img_bordes.to(device)

                out_tipo, out_defectos, out_llenado = model(img_color, img_bordes)

                tipo_pred_idx = torch.argmax(torch.softmax(out_tipo, dim=1), dim=1).item()
                tipo_pred_nombre = tipos_bebida_mapping_reverse[tipo_pred_idx]

                llenado_pred_idx = torch.argmax(torch.softmax(out_llenado, dim=1), dim=1).item()
                llenado_pred_nombre = llenado_mapping_reverse[llenado_pred_idx]

                defectos_probs = torch.sigmoid(out_defectos).squeeze().cpu().numpy()
                defectos_detectados = [
                    nombres_defectos[i] for i, prob in enumerate(defectos_probs) if prob > 0.5
                ]
                defectos_str = ", ".join(defectos_detectados) if defectos_detectados else "Ninguno"

                cv2.putText(frame, f"Tipo: {tipo_pred_nombre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Llenado: {llenado_pred_nombre}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Defectos: {defectos_str}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Guardar información de la imagén capturada
                img_frame_path = './img/'
                img_frame_name = img_frame_path + str(datetime.datetime.now()) + '.jpg'
                
                cv2.imwrite(img_frame_name, frame)

                botella = Botella()
                senial = False

                if defectos_str == "Ninguno":
                    botella.estado = 1
                else:
                    botella.estado = 0
                    senial = True

                botella.imagen = img_frame_name
                botella.ruta = img_frame_path
                
                if senial:
                    sendSerialData(ESTADO_BUENO, arduino)
                    time.sleep(TIEMPO_ESPERA)
                    sendSerialData(ESTADO_MALO, arduino)

                sendSerialData(botella)
            else:
                cv2.putText(frame, "Esperando botella...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

            cv2.imshow("YOLO + Clasificación", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")
    return 0

if __name__ == '__main__':    
    sys.exit(main(sys.argv))
