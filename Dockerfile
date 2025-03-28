# Utiliza una imagen base de Python
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para PyTorch
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean

# Configura el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de la aplicación al contenedor
COPY . /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto donde estará corriendo la API
EXPOSE 8003

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
