# modelo.py

import torch
import torch.nn as nn

class MultiInputModel(nn.Module):
    def __init__(self, num_tipos_bebida, num_defectos, num_llenado):
        super(MultiInputModel, self).__init__()

        # Dimensiones de salida después de Conv2d y MaxPool2d:
        # Input 224x224
        # Conv2d(kernel=3, padding=1) -> 224x224
        # MaxPool2d(kernel=2, stride=2) -> 112x112
        # Conv2d(kernel=3, padding=1) -> 112x112
        # MaxPool2d(kernel=2, stride=2) -> 56x56
        # -> Flatten -> 56*56 = 3136
        feature_map_dim = 56 * 56 # Dimensión de una feature map después de 2 MaxPool2d

        # Rama para la entrada de color (3 canales RGB)
        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Añadida una capa más
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Rama para la entrada de bordes (1 canal de escala de grises)
        self.bordes_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Añadida una capa más
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # El tamaño de la entrada a la capa FC fusionada será (número de filtros de color + número de filtros de bordes) * dimensión_final_de_feature_map
        self.fc_input_size = (128 + 128) * feature_map_dim # (128 + 128) * 56*56 = 256 * 3136 = 802816

        # Capas de fusión (ajustadas para manejar el tamaño de entrada)
        self.fc = nn.Linear(self.fc_input_size, 256) # Reducido a 256, puedes probar 512, 1024, etc.
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Añadido Dropout para regularización

        # Capas de salida
        self.fc_bebida = nn.Linear(256, num_tipos_bebida)
        self.fc_defectos = nn.Linear(256, num_defectos)
        self.fc_llenado = nn.Linear(256, num_llenado)

    def forward(self, input_color, input_bordes):
        color_features = self.color_branch(input_color)
        bordes_features = self.bordes_branch(input_bordes)

        # Concatenar las características de ambas ramas
        merged_features = torch.cat((color_features, bordes_features), dim=1)

        # Pasar por las capas de fusión
        merged_features = self.fc(merged_features)
        merged_features = self.relu(merged_features)
        merged_features = self.dropout(merged_features) # Aplicar dropout

        # Salidas finales (¡sin softmax/sigmoid aquí si se usa CrossEntropyLoss/BCEWithLogitsLoss!)
        bebida_output = self.fc_bebida(merged_features) # Logits para CrossEntropyLoss
        defectos_output = self.fc_defectos(merged_features) # Logits para BCEWithLogitsLoss
        llenado_output = self.fc_llenado(merged_features) # Logits para CrossEntropyLoss

        return bebida_output, defectos_output, llenado_output

# # Mover el modelo a la GPU si está disponible
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"\nUsando dispositivo: {device}")

# # Crear el modelo
# model = MultiInputModel(num_tipos_bebida, num_defectos, num_llenado)
# model.to(device)

# Opcional: Imprimir un resumen del modelo (necesita !pip install torchsummary)
# from torchsummary import summary
# summary(model, [(3, 224, 224), (1, 224, 224)], batch_size=batch_size)