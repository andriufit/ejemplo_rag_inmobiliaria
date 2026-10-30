# -*- coding: utf-8 -*-
"""
=============================================================================
 COMPARAR MODELOS DE EMBEDDING  -  ¿cual encuentra mejor los pisos?
=============================================================================
 Un modelo de embedding no es "bueno" o "malo" en abstracto: hay que medirlo
 con NUESTROS datos y NUESTRAS preguntas. Esto es una mini evaluacion:

   1. Preparamos preguntas en español y, para cada una, que piso(s) DEBERIAN
      salir (lo decide un humano: es el "ground truth").
   2. Para cada modelo: embeddings de los pisos y de las preguntas, y
      busqueda EXACTA por coseno (la misma de algoritmos.py, sin Qdrant,
      para comparar solo los modelos).
   3. Medimos:
      - Acierto@1: % de preguntas cuyo PRIMER resultado es correcto.
      - MRR (Mean Reciprocal Rank): media de 1/posicion del primer acierto.
        1.0 = siempre el primero; 0.5 = de media sale el segundo...

 Ejecutar (con los pisos de ejemplo cargados en MySQL):
     python comparar_embeddings.py
=============================================================================
"""
import time

import numpy as np
import ollama

import config
import db
from rag import piso_a_texto

_ollama = ollama.Client(host=config.OLLAMA_HOST)

# Modelo -> (prefijo para documentos, prefijo para consultas).
# nomic-embed-text esta entrenado con prefijos de tarea; los otros no los usan.
MODELOS = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "paraphrase-multilingual": ("", ""),
    "bge-m3": ("", ""),
}

# Pregunta -> IDs de los pisos que se consideran respuesta correcta
# (son los IDs que crea seed.py).
PREGUNTAS = {
    "casa con jardín y piscina": {5, 8},
    "cerca de la playa": {10},
    "para una familia con niños pequeños y jardín": {5, 8},
    "inversión barata para alquilar": {3},
    "zona con vida nocturna y bares": {1},
    "vivienda elegante con portero": {2},
    "rodeado de naturaleza y árboles": {11},
    "obra nueva con aire acondicionado": {7},
    "barrio de moda con restaurantes": {9},
    "edificio antiguo con balcones y vigas de madera": {6},
    "piso exterior con orientación sur cerca de colegios": {4},
}


def embeddings(modelo, textos):
    # Ollama acepta una lista de textos y devuelve un vector por texto.
    return np.array(_ollama.embed(model=modelo, input=textos).embeddings)


def evaluar(modelo, pisos):
    prefijo_doc, prefijo_consulta = MODELOS[modelo]
    inicio = time.time()

    docs = embeddings(modelo, [prefijo_doc + piso_a_texto(p) for p in pisos])
    consultas = embeddings(modelo, [prefijo_consulta + q for q in PREGUNTAS])

    # Normalizamos y multiplicamos matrices: cada celda es el COSENO entre una
    # consulta y un piso. Es la busqueda exacta de algoritmos.py, en bloque.
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    similitudes = consultas @ docs.T

    aciertos, rr, detalle = 0, [], []
    ids = [p["id"] for p in pisos]
    for fila, (pregunta, correctos) in zip(similitudes, PREGUNTAS.items()):
        ranking = [ids[i] for i in np.argsort(-fila)]   # IDs de mas a menos parecido
        posicion = next(i for i, pid in enumerate(ranking, 1) if pid in correctos)
        aciertos += posicion == 1
        rr.append(1 / posicion)
        detalle.append((pregunta, posicion, ranking[0]))

    return {
        "modelo": modelo,
        "dim": docs.shape[1],
        "acierto@1": aciertos / len(PREGUNTAS),
        "mrr": float(np.mean(rr)),
        "segundos": time.time() - inicio,
        "detalle": detalle,
    }


if __name__ == "__main__":
    pisos = db.listar_pisos()
    resultados = [evaluar(m, pisos) for m in MODELOS]

    print(f"\n{'modelo':26s} {'dim':>5s} {'acierto@1':>10s} {'MRR':>6s} {'tiempo':>8s}")
    for r in resultados:
        print(f"{r['modelo']:26s} {r['dim']:5d} {r['acierto@1']:10.0%} "
              f"{r['mrr']:6.2f} {r['segundos']:7.1f}s")

    # Detalle: en que posicion salio el piso correcto con cada modelo
    print(f"\n{'pregunta':52s}" + "".join(f"{m[:12]:>14s}" for m in MODELOS))
    for i, pregunta in enumerate(PREGUNTAS):
        print(f"{pregunta[:50]:52s}" +
              "".join(f"{'#' + str(r['detalle'][i][1]):>14s}" for r in resultados))
