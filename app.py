# -*- coding: utf-8 -*-
"""
=============================================================================
 API FLASK  -  punto de entrada de la aplicacion
=============================================================================
 Paginas:
   GET  /                      -> listado con filtros + chat con el agente
   GET  /piso/<id>             -> ficha de un inmueble
   GET  /admin                 -> alta / edicion / borrado de pisos

 API JSON:
   GET    /api/pisos           -> listado con filtros         (MySQL)
   GET    /api/pisos/<id>      -> un piso                     (MySQL)
   POST   /api/pisos           -> crear piso      (MySQL + indexar en Qdrant)
   PUT    /api/pisos/<id>      -> editar piso     (MySQL + reindexar en Qdrant)
   DELETE /api/pisos/<id>      -> borrar piso     (MySQL + borrar de Qdrant)
   GET    /api/buscar?q=...    -> busqueda semantica pura     (Qdrant)
   POST   /api/agente          -> conversar con el agente     (gemma3 + Qdrant)
   POST   /api/reindexar       -> reconstruir Qdrant desde MySQL

 Arrancar:  python app.py   ->  http://localhost:8000
=============================================================================
"""
import os

from flask import Flask, jsonify, render_template, request

import agente
import db
import rag

app = Flask(__name__)


# ---------------------------------------------------------------------------
# PAGINAS
# ---------------------------------------------------------------------------
@app.get("/")
def pagina_inicio():
    return render_template("index.html")


@app.get("/admin")
def pagina_admin():
    return render_template("admin.html")


@app.get("/piso/<int:piso_id>")
def pagina_piso(piso_id):
    # Ficha de cada inmueble: es a donde apuntan los enlaces del agente.
    piso = db.obtener_piso(piso_id)
    if not piso:
        return "Este inmueble ya no existe", 404
    return render_template("piso.html", piso=piso)


# ---------------------------------------------------------------------------
# LECTURA: filtros clasicos -> SQL
# ---------------------------------------------------------------------------
def _bool_param(nombre):
    # Convierte ?ascensor=1 / ?ascensor=0 en True/False; si no viene, None.
    valor = request.args.get(nombre)
    return None if valor in (None, "") else valor == "1"


@app.get("/api/pisos")
def api_listar():
    pisos = db.listar_pisos(
        ciudad=request.args.get("ciudad"),
        tipo=request.args.get("tipo"),
        precio_max=request.args.get("precio_max"),
        metros_min=request.args.get("metros_min"),
        ascensor=_bool_param("ascensor"),
        luminoso=_bool_param("luminoso"),
    )
    return jsonify(pisos)


@app.get("/api/pisos/<int:piso_id>")
def api_obtener(piso_id):
    piso = db.obtener_piso(piso_id)
    return jsonify(piso) if piso else (jsonify({"error": "No existe"}), 404)


# ---------------------------------------------------------------------------
# ESCRITURA: aqui es donde se ve la SINCRONIZACION MySQL -> Qdrant
# ---------------------------------------------------------------------------
# Patron usado: "escritura doble sincrona".
#   1) Se guarda en MySQL (fuente de la verdad).
#   2) Se lee la fila guardada y se (re)indexa en Qdrant en la misma peticion.
# Ventaja: es simple y el cambio se ve al instante en el buscador.
# Limitacion: si Qdrant falla despues de guardar en MySQL, quedan
# desincronizados. En produccion se suele resolver con una COLA (Celery,
# RabbitMQ...) o un proceso que reindexa los pisos con 'actualizado' reciente.
# Para arreglarlo a mano tenemos POST /api/reindexar.

def _datos_formulario():
    d = request.get_json()
    return {
        "titulo": d["titulo"],
        "ciudad": d["ciudad"],
        "barrio": d["barrio"],
        "tipo": d["tipo"],
        # tipo_chalet solo aplica a chalets
        "tipo_chalet": (d.get("tipo_chalet") or None) if d["tipo"] == "chalet" else None,
        "luminoso": bool(d.get("luminoso")),
        "ascensor": bool(d.get("ascensor")),
        "precio": int(d["precio"]),
        "metros": int(d["metros"]),
        "descripcion": d["descripcion"],
    }


@app.post("/api/pisos")
def api_crear():
    piso_id = db.crear_piso(_datos_formulario())      # 1) MySQL
    piso = db.obtener_piso(piso_id)
    rag.indexar_piso(piso)                            # 2) Qdrant
    return jsonify(piso), 201


@app.put("/api/pisos/<int:piso_id>")
def api_editar(piso_id):
    db.actualizar_piso(piso_id, _datos_formulario())  # 1) MySQL
    piso = db.obtener_piso(piso_id)
    rag.indexar_piso(piso)                            # 2) Qdrant (upsert = sobreescribe)
    return jsonify(piso)


@app.delete("/api/pisos/<int:piso_id>")
def api_borrar(piso_id):
    db.borrar_piso(piso_id)                           # 1) MySQL
    rag.desindexar_piso(piso_id)                      # 2) Qdrant
    return jsonify({"ok": True})


@app.post("/api/reindexar")
def api_reindexar():
    return jsonify({"indexados": rag.reindexar_todo()})


# ---------------------------------------------------------------------------
# RAG y AGENTE
# ---------------------------------------------------------------------------
@app.get("/api/buscar")
def api_buscar():
    # Solo la "R" de RAG: sin LLM. Util para ver que devuelve Qdrant y con
    # que score (similitud del coseno) antes de que el agente lo interprete.
    return jsonify(rag.buscar(request.args.get("q", ""),
                              ciudad=request.args.get("ciudad") or None))


@app.post("/api/agente")
def api_agente():
    # El navegador solo manda su ID de conversacion y el mensaje nuevo:
    # el historial y el perfil (la MEMORIA) los guarda el servidor.
    datos = request.get_json()
    try:
        respuesta, perfil = agente.preguntar(
            datos["conversacion_id"], datos["mensaje"], request.host_url)
    except Exception as error:  # p.ej. Ollama apagado o modelo no descargado
        return jsonify({"respuesta": f"Error llamando al LLM: {error}",
                        "memoria": None}), 500
    return jsonify({"respuesta": respuesta, "memoria": perfil})


if __name__ == "__main__":
    # El 5000 tipico de Flask suele estar reservado en Windows; usamos el 8000.
    # En Docker HOST=0.0.0.0 para que se pueda acceder desde fuera del contenedor.
    app.run(host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")), debug=True)
