# -*- coding: utf-8 -*-
"""
Carga inicial de datos: crea la tabla en MySQL, inserta 11 viviendas y
construye el indice vectorial en Qdrant.

Ejecutar una vez (o cuando quieras volver al estado inicial):
    python seed.py

Con --si-vacio solo carga datos si la tabla esta vacia (lo usa Docker al
arrancar, para no borrar lo que se haya creado desde el admin).
"""
import sys

import db
import rag

PISOS = [
    # ----------------------------- MADRID (5) -----------------------------
    dict(titulo="Ático con terraza en Malasaña", ciudad="Madrid", barrio="Malasaña",
         tipo="piso", tipo_chalet=None, luminoso=True, ascensor=True,
         precio=395000, metros=62,
         descripcion="Ático reformado con terraza de 15 m2 y vistas a los tejados. "
                     "Zona con mucho ambiente, bares y vida nocturna; puede ser ruidoso los fines de semana."),
    dict(titulo="Piso señorial en Salamanca", ciudad="Madrid", barrio="Salamanca",
         tipo="piso", tipo_chalet=None, luminoso=True, ascensor=True,
         precio=890000, metros=140,
         descripcion="Finca clásica con portero, techos altos y suelos de madera. "
                     "Cuatro habitaciones, ideal para familia. Zona elegante y segura."),
    dict(titulo="Interior económico en Vallecas", ciudad="Madrid", barrio="Puente de Vallecas",
         tipo="piso", tipo_chalet=None, luminoso=False, ascensor=False,
         precio=165000, metros=55,
         descripcion="Tercero sin ascensor, interior y a reformar. Buena opción de inversión "
                     "para alquilar; bien comunicado con metro."),
    dict(titulo="Piso familiar en Chamberí", ciudad="Madrid", barrio="Chamberí",
         tipo="piso", tipo_chalet=None, luminoso=True, ascensor=True,
         precio=560000, metros=95,
         descripcion="Exterior a calle tranquila, tres habitaciones, colegios y mercado cerca. "
                     "Muy buena orientación sur."),
    dict(titulo="Chalet pareado en Las Rozas", ciudad="Madrid", barrio="Las Rozas",
         tipo="chalet", tipo_chalet="pareado", luminoso=True, ascensor=False,
         precio=620000, metros=210,
         descripcion="Chalet pareado con jardín privado y piscina comunitaria. "
                     "Urbanización tranquila a 25 minutos de Madrid en coche; ideal para teletrabajar."),
    # ---------------------------- BARCELONA (3) ----------------------------
    dict(titulo="Piso con encanto en Gràcia", ciudad="Barcelona", barrio="Gràcia",
         tipo="piso", tipo_chalet=None, luminoso=True, ascensor=False,
         precio=345000, metros=68,
         descripcion="Finca antigua sin ascensor con balcones y vigas de madera. "
                     "Barrio con plazas, ambiente de pueblo y comercio local."),
    dict(titulo="Piso moderno en el Eixample", ciudad="Barcelona", barrio="Eixample",
         tipo="piso", tipo_chalet=None, luminoso=False, ascensor=True,
         precio=480000, metros=90,
         descripcion="Obra nueva con aire acondicionado y domótica. Da a patio de manzana: "
                     "silencioso pero con poca luz directa."),
    dict(titulo="Chalet adosado en Sant Cugat", ciudad="Barcelona", barrio="Sant Cugat del Vallès",
         tipo="chalet", tipo_chalet="adosado", luminoso=True, ascensor=False,
         precio=750000, metros=230,
         descripcion="Adosado de tres plantas con jardín y garaje para dos coches. "
                     "Colegios internacionales cerca; entorno verde y familiar."),
    # ----------------------------- VALENCIA (3) ----------------------------
    dict(titulo="Piso reformado en Ruzafa", ciudad="Valencia", barrio="Ruzafa",
         tipo="piso", tipo_chalet=None, luminoso=True, ascensor=True,
         precio=285000, metros=85,
         descripcion="Reforma integral, cocina abierta y dos baños. Barrio de moda con "
                     "restaurantes y galerías."),
    dict(titulo="Casa de pueblo en el Cabanyal", ciudad="Valencia", barrio="El Cabanyal",
         tipo="piso", tipo_chalet=None, luminoso=True, ascensor=False,
         precio=210000, metros=75,
         descripcion="Planta baja con fachada de azulejos a cinco minutos andando de la playa "
                     "de la Malvarrosa. Ambiente marinero y tranquilo."),
    dict(titulo="Chalet pareado en Godella", ciudad="Valencia", barrio="Godella",
         tipo="chalet", tipo_chalet="pareado", luminoso=False, ascensor=False,
         precio=390000, metros=180,
         descripcion="Pareado rodeado de pinos, muy fresco en verano pero con poca luz en invierno. "
                     "Metro a Valencia en 20 minutos."),
]

if __name__ == "__main__":
    print("1) Creando esquema en MySQL...")
    db.crear_esquema()

    if "--si-vacio" in sys.argv and db.listar_pisos():
        print("Ya hay datos: no se recarga nada.")
        sys.exit(0)

    print("2) Insertando pisos en MySQL...")
    with db.conectar() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE pisos")  # empezamos de cero
    for piso in PISOS:
        db.crear_piso(piso)

    print("3) Generando embeddings e indexando en Qdrant...")
    total = rag.reindexar_todo()
    print(f"Listo: {total} pisos en MySQL y en Qdrant.")
