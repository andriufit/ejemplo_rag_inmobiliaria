# -*- coding: utf-8 -*-
"""
=============================================================================
 CAPA VECTORIAL (Qdrant + embeddings)  ->  la "R" de RAG (Retrieval)
=============================================================================
 Lo que en algoritmos.py haciamos a mano (calcular el coseno contra todos
 los vectores y ordenar) aqui lo hace Qdrant por nosotros, y ademas con un
 indice HNSW para que escale a millones de vectores.

 Flujo de INDEXADO (cuando se crea o edita un piso):
     fila MySQL -> texto descriptivo -> embedding (768 numeros, Ollama) -> Qdrant

 Flujo de BUSQUEDA (cuando el usuario pregunta):
     pregunta -> embedding -> Qdrant busca los vectores mas cercanos
              -> devuelve IDs -> leemos los datos frescos de MySQL

 Clave de diseño: el ID del punto en Qdrant ES el mismo que el ID de la fila
 en MySQL. Esa es la "costura" entre las dos bases de datos.
=============================================================================
"""
import ollama
from qdrant_client import QdrantClient, models

import config
import db

_ollama = ollama.Client(host=config.OLLAMA_HOST)
_qdrant = QdrantClient(url=config.QDRANT_URL)


# ---------------------------------------------------------------------------
# 1. DE FILA DE BASE DE DATOS A TEXTO
# ---------------------------------------------------------------------------
def piso_a_texto(piso):
    # El embedding se calcula sobre TEXTO, no sobre columnas. Por eso
    # "contamos" el piso en lenguaje natural, mezclando datos estructurados
    # y la descripcion libre. Lo que no este en este texto, el buscador
    # semantico NO lo puede "ver".
    tipo = piso["tipo"]
    if piso["tipo"] == "chalet" and piso.get("tipo_chalet"):
        tipo = f"chalet {piso['tipo_chalet']}"
    partes = [
        f"{tipo.capitalize()} en {piso['ciudad']}, barrio {piso['barrio']}.",
        f"{piso['metros']} metros cuadrados por {piso['precio']} euros.",
        "Muy luminoso, con mucha luz natural." if piso["luminoso"]
        else "Poco luminoso, interior.",
        "Tiene ascensor." if piso["ascensor"] else "Sin ascensor.",
        piso["descripcion"],
    ]
    return " ".join(partes)


def embedding(texto, es_consulta=False):
    # Algunos modelos (nomic-embed-text) estan entrenados con PREFIJOS de
    # tarea que distinguen el documento que se guarda de la pregunta con la
    # que se busca. Es un detalle propio de cada modelo: el resto no los usa.
    prefijo = ""
    if config.EMBEDDING_MODEL.startswith("nomic"):
        prefijo = "search_query: " if es_consulta else "search_document: "
    respuesta = _ollama.embed(model=config.EMBEDDING_MODEL, input=prefijo + texto)
    return respuesta.embeddings[0]  # lista de EMBEDDING_DIM floats


# ---------------------------------------------------------------------------
# 2. PREPARAR LA COLECCION (equivale a un CREATE TABLE en Qdrant)
# ---------------------------------------------------------------------------
def crear_coleccion(reiniciar=False):
    if reiniciar and _qdrant.collection_exists(config.QDRANT_COLLECTION):
        _qdrant.delete_collection(config.QDRANT_COLLECTION)
    if not _qdrant.collection_exists(config.QDRANT_COLLECTION):
        _qdrant.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            # COSINE: la misma metrica que vimos en algoritmos.py, la
            # habitual para texto (mide direccion = significado).
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_DIM, distance=models.Distance.COSINE),
        )


# ---------------------------------------------------------------------------
# 3. SINCRONIZACION MySQL -> Qdrant
# ---------------------------------------------------------------------------
def indexar_piso(piso):
    # "upsert" = insert o update: si el ID ya existe en Qdrant se SOBREESCRIBE.
    # Por eso sirve igual para pisos nuevos y para pisos editados.
    _qdrant.upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=[models.PointStruct(
            id=piso["id"],                       # mismo ID que en MySQL
            vector=embedding(piso_a_texto(piso)),
            # El PAYLOAD son metadatos que viajan con el vector. Guardamos
            # unos pocos para poder FILTRAR dentro de Qdrant (ver buscar()).
            # No guardamos todo: el dato oficial sigue estando en MySQL.
            payload={
                "ciudad": piso["ciudad"],
                "tipo": piso["tipo"],
                "precio": piso["precio"],
                "metros": piso["metros"],
                "texto": piso_a_texto(piso),     # util para depurar
            },
        )],
    )


def desindexar_piso(piso_id):
    _qdrant.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.PointIdsList(points=[piso_id]),
    )


def reindexar_todo():
    # Reconstruye el indice vectorial entero a partir de MySQL.
    # En un proyecto real esto se lanza tras cambiar de modelo de embedding
    # o si se sospecha que Qdrant y MySQL se han desincronizado.
    crear_coleccion(reiniciar=True)
    pisos = db.listar_pisos()
    for piso in pisos:
        indexar_piso(piso)
    return len(pisos)


# ---------------------------------------------------------------------------
# 4. BUSQUEDA SEMANTICA
# ---------------------------------------------------------------------------
def buscar(consulta, ciudad=None, precio_max=None, metros_min=None, limite=5):
    # BUSQUEDA HIBRIDA: semantica (vector) + filtro exacto (payload).
    # Qdrant aplica el filtro DURANTE la busqueda, no despues, asi que
    # siempre devuelve 'limite' resultados que cumplen el filtro.
    condiciones = []
    if ciudad:
        condiciones.append(models.FieldCondition(
            key="ciudad", match=models.MatchValue(value=ciudad)))
    if precio_max:
        condiciones.append(models.FieldCondition(
            key="precio", range=models.Range(lte=int(precio_max))))
    if metros_min:
        condiciones.append(models.FieldCondition(
            key="metros", range=models.Range(gte=int(metros_min))))

    respuesta = _qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embedding(consulta, es_consulta=True),
        query_filter=models.Filter(must=condiciones) if condiciones else None,
        limit=limite,
    )

    # Qdrant nos da (id, score). Los datos completos y actualizados los
    # pedimos a MySQL, y respetamos el orden de relevancia de Qdrant.
    scores = {p.id: p.score for p in respuesta.points}
    pisos = db.obtener_pisos_por_ids(list(scores))
    for piso in pisos:
        piso["score"] = round(scores[piso["id"]], 4)  # similitud del coseno
    return sorted(pisos, key=lambda p: p["score"], reverse=True)
