FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar requerimientos
COPY IA_Proyecto/notebooks/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar dataset base
COPY Telco.csv /app/Telco.csv

# Copiar el resto del proyecto
COPY IA_Proyecto /app/IA_Proyecto

WORKDIR /app/IA_Proyecto/notebooks
EXPOSE 8888

CMD ["python", "pipeline.py"]