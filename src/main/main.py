import os
import sys
import serial
import requests

from dto import Botella, toJSONBotella
from dotenv import load_dotenv

ESTADO_BUENO = "0"
ESTADO_MALO = "1"

"""
    Función encargada de abrir una conexión serie con arduino, pedendiedo el sistema operativo
    empleada, para enviar datos se emplea de la siguiente manera:

    1. Abrir una conexión
        conn = serialConnect()

    2. Enviar mensajes
        sendSerialData(ESTADO_BUENO, conn)
"""
def serialConnect() -> None:
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


def main(args) -> int:
    # TODO: Aquí es donde se inicia todo
    return 0

if __name__ == '__main__':    
    sys.exit(main(sys.argv))
