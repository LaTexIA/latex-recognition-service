from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Latex Recognition Service")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer los datos de la imagen subida
    image_data = await file.read()
    
    # Placeholder: Código LaTeX de prueba (reemplazar con lógica real si es necesario)
    latex_code = r"\frac{a}{b} = c"
    
    # Devolver el código LaTeX en formato JSON
    return JSONResponse(content={"latex_code": latex_code})