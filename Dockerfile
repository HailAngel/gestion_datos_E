FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar e instalar requerimientos aprovechando la caché de capas de Docker
COPY IA_Proyecto/notebooks/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el dataset original respetando la estructura de rutas del script (../../Telco.csv)
COPY Telco.csv /app/Telco.csv

# Copiar el resto del proyecto de desarrollo
COPY IA_Proyecto /app/IA_Proyecto

# Establecer el directorio de trabajo donde están tus scripts y notebooks
WORKDIR /app/IA_Proyecto/notebooks

# Exponer puerto estándar para Jupyter Lab/Notebook
EXPOSE 8888

# Comando por defecto: Ejecutar el pipeline automatizado
CMD ["python", "pipeline.py"]