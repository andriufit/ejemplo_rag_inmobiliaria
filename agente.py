# -*- coding: utf-8 -*-
"""
=============================================================================
 AGENTE ASESOR INMOBILIARIO  ->  la "G" de RAG (Generation) + memoria
=============================================================================
 El agente funciona como lo haria un comercial de verdad:

   1. ENTREVISTA: antes de buscar nada pregunta lo minimo imprescindible:
        - ciudad
        - tamaño (metros cuadrados minimos)
        - una pregunta LIBRE: "¿que otras cosas consideras importantes?"
      Lo que va sabiendo lo guarda en un PERFIL del cliente (su memoria).

   2. BUSQUEDA HIBRIDA en Qdrant:
        - la respuesta libre se convierte en embedding -> busqueda por significado
        - ciudad y metros se aplican como filtros EXACTOS sobre el payload

   3. RESPUESTA: Qdrant NO devuelve informacion "legible": devuelve IDs y
      puntuaciones de similitud (numeros). Un embedding no es informacion
      literal, es una posicion en un espacio de significados. Por eso:
        IDs de Qdrant -> datos reales de MySQL -> texto + enlace a la ficha
        -> el LLM redacta la recomendacion final en lenguaje natural.
      El usuario nunca ve JSON, IDs ni scores.
=============================================================================
"""
import re

import ollama

import config
import rag

_ollama = ollama.Client(host=config.OLLAMA_HOST)

CIUDADES = ["Madrid", "Barcelona", "Valencia"]

# Las preguntas de la entrevista, en orden. Son textos FIJOS: para preguntar
# no hace falta un LLM (es mas rapido, gratis y siempre sale igual).
PREGUNTAS = {
    "ciudad": "¿En qué ciudad buscas? Tenemos viviendas en Madrid, Barcelona "
              "y Valencia (o dime «me da igual»).",
    "metros_min": "¿Qué tamaño necesitas, más o menos? Dime los metros "
                  "cuadrados mínimos (o «me da igual»).",
    "preferencias": "¿Qué otras cosas consideras importantes a la hora de "
                    "buscar? Por ejemplo: luz, ascensor, jardín, tranquilidad, "
                    "cerca del mar, presupuesto...",
}

PROMPT_ASESOR = """Eres un asesor inmobiliario honesto que ayuda a decidir que \
vivienda comprar. Responde en español, en texto plano, breve y cercano.

Reglas:
- Usa UNICAMENTE las viviendas de la lista de RESULTADOS; no inventes datos.
- Recomienda como mucho 3, de mejor a peor encaje con lo que pide el cliente,
  con sus pros y contras reales (precio, precio por m2, luz, ascensor, barrio).
- Despues de cada vivienda que menciones, escribe su enlace EXACTAMENTE como
  aparece en la lista.
- No hables de identificadores, puntuaciones, bases de datos ni JSON.
- Si ninguna encaja, dilo claramente: tambien es valido recomendar NO comprar."""


# ---------------------------------------------------------------------------
# 1. MEMORIA
# ---------------------------------------------------------------------------
# Guardamos cada conversacion en un diccionario en RAM, indexado por un ID
# que genera el navegador. Contiene:
#   - historial: los mensajes (memoria "conversacional")
#   - perfil:    lo que sabemos del cliente (memoria "estructurada")
#   - pendiente: que pregunta le hicimos la ultima vez
# LIMITACION: se pierde al reiniciar Flask y no se comparte entre servidores.
# En produccion iria en Redis o en una tabla de la base de datos.
_memoria = {}


def obtener_memoria(conversacion_id):
    return _memoria.setdefault(conversacion_id, {
        "historial": [],
        "perfil": {"ciudad": None, "metros_min": None, "preferencias": None},
        "pendiente": None,
    })


# ---------------------------------------------------------------------------
# 2. ENTENDER LAS RESPUESTAS (sin LLM)
# ---------------------------------------------------------------------------
# Ciudad y metros son datos CERRADOS: un par de reglas sencillas los
# extraen de forma mas fiable que un LLM pequeño (que a veces se inventa
# ciudades). Regla practica: usa un LLM solo donde aporte algo.
INDIFERENTE = ("igual", "cualquier", "indiferente", "todas", "no importa")


def extraer_ciudad(texto, es_respuesta):
    t = texto.lower()
    for ciudad in CIUDADES:
        if ciudad.lower() in t:
            return ciudad
    if es_respuesta and any(p in t for p in INDIFERENTE):
        return "cualquiera"
    return None


def extraer_metros(texto, es_respuesta):
    # "90 m2", "90 metros", "unos 90m²"... y si es la respuesta a la
    # pregunta de tamaño, vale tambien un numero suelto ("90").
    m = re.search(r"(\d{2,4})\s*(m2|m²|metros|m\b)", texto.lower())
    if m:
        return int(m.group(1))
    if es_respuesta:
        m = re.search(r"\d{2,4}", texto)
        if m:
            return int(m.group())
        if any(p in texto.lower() for p in INDIFERENTE):
            return 0  # 0 = sin minimo
    return None


def actualizar_perfil(memoria, mensaje):
    perfil, pendiente = memoria["perfil"], memoria["pendiente"]

    ciudad = extraer_ciudad(mensaje, es_respuesta=pendiente == "ciudad")
    if ciudad:
        perfil["ciudad"] = ciudad

    metros = extraer_metros(mensaje, es_respuesta=pendiente == "metros_min")
    if metros is not None:
        perfil["metros_min"] = metros

    # La pregunta libre se guarda tal cual: sera el texto de la busqueda
    # semantica (lo que se convierte en embedding).
    if pendiente == "preferencias":
        perfil["preferencias"] = mensaje


# ---------------------------------------------------------------------------
# 3. DE RESULTADO DE QDRANT A INFORMACION LEGIBLE
# ---------------------------------------------------------------------------
def euros(numero):
    return f"{numero:,} €".replace(",", ".")  # a la española: 395.000 €


def piso_a_linea(piso, base_url):
    # Convertimos cada resultado en una linea de texto que entiende una
    # persona (y el LLM). Fijate en lo que NO incluimos: ni id ni score.
    tipo = piso["tipo"] + (f" {piso['tipo_chalet']}" if piso["tipo_chalet"] else "")
    return (
        f"- {piso['titulo']} ({tipo} en {piso['barrio']}, {piso['ciudad']}): "
        f"{piso['metros']} m2, {euros(piso['precio'])} "
        f"({euros(piso['precio'] // piso['metros'])}/m2), "
        f"{'luminoso' if piso['luminoso'] else 'poco luminoso'}, "
        f"{'con ascensor' if piso['ascensor'] else 'sin ascensor'}. "
        f"{piso['descripcion']} Enlace: {base_url}piso/{piso['id']}"
    )


# ---------------------------------------------------------------------------
# 4. EL FLUJO COMPLETO
# ---------------------------------------------------------------------------
def preguntar(conversacion_id, mensaje, base_url):
    """
    Recibe el mensaje del usuario y devuelve (respuesta_en_texto, perfil).
    base_url: p.ej. "http://localhost:8000/", para construir los enlaces.
    """
    memoria = obtener_memoria(conversacion_id)
    memoria["historial"].append({"role": "user", "content": mensaje})
    actualizar_perfil(memoria, mensaje)
    perfil = memoria["perfil"]

    # ¿Falta algun dato? -> preguntamos el siguiente y NO buscamos todavia.
    falta = next((dato for dato, valor in perfil.items() if valor is None), None)
    if falta:
        memoria["pendiente"] = falta
        respuesta = PREGUNTAS[falta]
        if len(memoria["historial"]) == 1:  # primer mensaje: nos presentamos
            respuesta = ("¡Hola! Para recomendarte bien necesito un par de "
                         "datos. " + respuesta)
    else:
        memoria["pendiente"] = None
        respuesta = buscar_y_responder(memoria, mensaje, base_url)

    memoria["historial"].append({"role": "assistant", "content": respuesta})
    return respuesta, perfil


def buscar_y_responder(memoria, mensaje, base_url):
    perfil = memoria["perfil"]

    # La consulta semantica son las preferencias del cliente; si ya estamos
    # en una pregunta de seguimiento ("¿y alguno con ascensor?"), la añadimos.
    consulta = perfil["preferencias"]
    if mensaje != perfil["preferencias"]:
        consulta += ". " + mensaje

    pisos = rag.buscar(
        consulta,
        ciudad=None if perfil["ciudad"] == "cualquiera" else perfil["ciudad"],
        metros_min=perfil["metros_min"] or None,
        limite=4,
    )

    # Esto es lo que devuelve la busqueda "por dentro": solo lo vemos en la
    # consola del servidor, nunca le llega al usuario.
    print(f"[agente] consulta={consulta!r} perfil={perfil} -> "
          f"{[(p['id'], p['score']) for p in pisos]}")

    if pisos:
        resultados = "\n".join(piso_a_linea(p, base_url) for p in pisos)
    else:
        resultados = "No hay ninguna vivienda que cumpla la ciudad y el tamaño pedidos."

    # Mandamos al LLM: instrucciones + los ultimos mensajes (memoria) +
    # el perfil y los resultados ya convertidos a texto.
    mensajes = (
        [{"role": "system", "content": PROMPT_ASESOR}]
        + memoria["historial"][-10:]  # limitamos para no llenar el contexto
        + [{"role": "system", "content":
            f"PERFIL DEL CLIENTE: ciudad={perfil['ciudad']}, "
            f"metros minimos={perfil['metros_min'] or 'sin minimo'}, "
            f"le importa: {perfil['preferencias']}\n\nRESULTADOS:\n{resultados}"}]
    )
    texto = _ollama.chat(model=config.LLM_MODEL, messages=mensajes).message.content
    # Aunque se le pida texto plano, el LLM tiende a usar markdown (**negrita**).
    # Limpiar la salida del modelo con codigo es habitual y mas fiable que
    # insistir en el prompt.
    return texto.replace("**", "").strip()
