import serial
import os

from dotenv import load_dotenv

ESTADO_BUENO = "0"
ESTADO_MALO = "1"

def conectar() -> None:
    load_dotenv()

    port = os.getenv('SERIAL_PORT')
    arduino = serial.Serial(port, 9600)
    return arduino

def enviar(msg: str, serial) -> None:
    serial.write(msg.encode())

conn = conectar()

# Mensaje si una botella esta buena
enviar(ESTADO_BUENO, conn)

# Mensaje si una botella esta mala
enviar(ESTADO_MALO, conn)
