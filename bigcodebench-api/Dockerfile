# Reutiliza la imagen oficial de BigCodeBench: ya trae el paquete `bigcodebench`
# (con el dataset oficial y el sandbox de ejecución) y sus dependencias pesadas.
# Solo le agregamos una API HTTP encima; el entrypoint original se reemplaza.
FROM bigcodebench/bigcodebench-evaluate:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
