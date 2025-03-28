import torch
import torch.nn as nn
import os
from torchvision import datasets
import pickle

# Define la clase BetterConvNet
import torch.nn as nn

class BetterConvNet(nn.Module):
    def __init__(self, num_classes):
        super(BetterConvNet, self).__init__()
        
        # Parte de extracción de características (ajustada para 52x52)
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 52x52 -> 52x52
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 52x52 -> 26x26
            
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 26x26 -> 26x26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 26x26 -> 13x13
            
            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 13x13 -> 13x13
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 13x13 -> 6x6 (redondeo hacia abajo)
            
            # Bloque 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 6x6 -> 6x6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 6x6 -> 3x3
        )
        
        # Pooling adaptativo global
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 3x3 -> 1x1
        
        # Clasificador (igual que antes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)          # Extrae características: [batch, 256, 3, 3]
        x = self.avgpool(x)           # Reduce a [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)     # Aplana a [batch, 256]
        x = self.classifier(x)        # Clasifica: [batch, num_classes]
        return x

class_names = None
# Para cargar las clases desde el archivo:
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

num_classes = len(class_names)

# Carga del modelo entrenado
model = BetterConvNet(num_classes=82)

# Carga los pesos entrenados
model_path = 'model.pth'  # <- Ruta al archivo de pesos entrenado
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Establece el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Modo evaluación
