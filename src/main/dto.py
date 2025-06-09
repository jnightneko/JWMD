import datetime

class Botella:
    def __init__(self):
        self.id = 0
        self.imagen = ''
        self.ruta = ''
        self.descripcion = ''
        self.estado = 0
        self.fecha = datetime.datetime.now()

def toJSONBotella(botella: Botella):
    return {
        'imagen': botella.imagen,
        'ruta': botella.ruta,
        'descripcion': botella.descripcion,
        'estado': botella.estado,
        'fecha': botella.fecha
    }
