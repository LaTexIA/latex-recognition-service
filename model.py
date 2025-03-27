import torch
import torch.nn as nn
import os
from torchvision import datasets

# Define la clase BetterConvNet
class BetterConvNet(nn.Module):
    def __init__(self, num_classes):
        super(BetterConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Ruta a las clases (dataset para saber el número de clases y los nombres)
train_data_path = 'data/train'  # <- Modifica si tu carpeta está en otra ruta

# Cargamos el dataset (solo para obtener los nombres de clases)
train_dataset = datasets.ImageFolder(root=train_data_path)
class_names = train_dataset.classes
num_classes = len(class_names)

# Carga del modelo entrenado
model = BetterConvNet(num_classes=num_classes)

# Carga los pesos entrenados
model_path = 'model.pth'  # <- Ruta al archivo de pesos entrenado
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Establece el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Modo evaluación
