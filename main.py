from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
import json

app = FastAPI(title="Latex Recognition Service")

# Cargar el modelo entrenado
try:
    model = torch.jit.load('src/infrastructure/model/model.pt')
    model.eval()  # Configurar el modelo en modo evaluación
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.post("/predict")
async def predict(request: Request):
    """
    Recibe un arreglo NumPy serializado en JSON con la imagen preprocesada,
    lo deserializa y usa el modelo para predecir el código LaTeX.
    Devuelve el resultado en formato JSON.
    """
    try:
        # Leer el cuerpo de la solicitud como JSON
        data = await request.json()
        
        # Deserializar el arreglo NumPy desde JSON
        image_array = np.array(json.loads(data['image_array']), dtype=np.float32)
        
        # Asegurarse de que el arreglo tenga la forma correcta para el modelo
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Convertir el arreglo NumPy a un tensor de PyTorch
        input_tensor = torch.from_numpy(image_array).unsqueeze(0)
        
        # Realizar la predicción con el modelo
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Suponiendo que el modelo devuelve el código LaTeX como una cadena
        latex_code = prediction.argmax(dim=1).item()  # Ajusta esto según la salida real de tu modelo
        
        # Devolver el resultado en formato JSON
        return JSONResponse(content={"latex_code": latex_code})
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid input format. 'image_array' key is required.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")