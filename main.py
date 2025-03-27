from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageChops
import torchvision.transforms as transforms
import torch
import io

# Modelo cargado
from model import model, class_names, device  # Asegúrate de tener tu modelo ya entrenado cargado
from predict import predict_image_tensor  # Función que definiremos aparte

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer imagen desde el archivo recibido
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # PREPROCESAMIENTO basado en PIL y ImageChops
    white_image = Image.new('RGB', image.size, (255, 255, 255))
    diff = ImageChops.difference(image, white_image)
    bbox = diff.getbbox()
    
    if bbox is None:
        return JSONResponse(content={"error": "La imagen parece estar vacía"}, status_code=400)

    image = image.crop(bbox)
    image = image.resize((128, 128))

    # Transforma a tensor normalizado
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    # Predicción
    predicted_index, predicted_class = predict_image_tensor(model, input_tensor, class_names)

    return {"predicted_class": predicted_class, "class_index": predicted_index}
