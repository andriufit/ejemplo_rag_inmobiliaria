# -*- coding: utf-8 -*-
"""
Configuracion centralizada del proyecto.

Todo lo que cambia entre entornos (tu portatil, el de un alumno, produccion)
se lee de variables de entorno / fichero .env. Asi el codigo no cambia.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # lee el fichero .env si existe

# --- Base de datos RELACIONAL (MySQL/MariaDB de XAMPP) -----------------------
# Es la FUENTE DE LA VERDAD: aqui viven los datos "oficiales" de cada piso.
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "inmobiliaria")

# --- Base de datos VECTORIAL (Qdrant) ----------------------------------------
# Es un INDICE DERIVADO: guarda el embedding de cada piso para buscar por
# significado. Si se borrara entero, se podria reconstruir desde MySQL.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pisos")

# --- OLLAMA (modelos en local) ------------------------------------------------
# Todo corre en tu maquina: ni API keys ni datos saliendo a internet.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Modelo de EMBEDDING: convierte texto en un vector de numeros.
# Elegido con comparar_embeddings.py: con preguntas en español,
# paraphrase-multilingual acierta el 82% a la primera frente al 36% de
# nomic-embed-text (entrenado sobre todo en ingles).
# Si cambias de modelo: ajusta EMBEDDING_DIM y REINDEXA (python seed.py),
# porque los vectores de modelos distintos no son comparables entre si.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # bge-m3 seria 1024

# Modelo de LENGUAJE del agente (el que "piensa" y redacta).
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3")
