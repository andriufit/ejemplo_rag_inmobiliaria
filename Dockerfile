# Imagen de la app Flask. Las bases de datos (MariaDB y Qdrant) van en sus
# propios contenedores: ver docker-compose.yml.
FROM python:3.13-slim

WORKDIR /app

# Primero solo las dependencias: asi Docker cachea esta capa y no reinstala
# todo cada vez que cambias una linea de codigo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 1) Carga los datos de ejemplo si la base de datos esta vacia. Se reintenta
#    porque MariaDB/Qdrant pueden tardar unos segundos en estar listos.
# 2) Arranca la API.
CMD ["sh", "-c", "until python seed.py --si-vacio; do echo 'Esperando a MySQL/Qdrant...'; sleep 3; done && exec python app.py"]
