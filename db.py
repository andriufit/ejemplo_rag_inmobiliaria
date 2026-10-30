# -*- coding: utf-8 -*-
"""
=============================================================================
 CAPA RELACIONAL (MySQL / MariaDB)
=============================================================================
 Aqui solo hay SQL "de toda la vida". Esta base de datos es la fuente de la
 verdad: precios, metros, si tiene ascensor... son datos EXACTOS y se
 filtran con WHERE, no con embeddings.

 Regla de oro de un proyecto real con RAG:
   - Filtros exactos (precio <= 300000, ciudad = 'Madrid')  -> SQL
   - Busqueda por significado ("algo tranquilo y con luz")  -> Qdrant
=============================================================================
"""
import pymysql
import config


def conectar(con_db=True):
    # DictCursor hace que cada fila sea un dict {"columna": valor},
    # que luego se convierte a JSON directamente.
    return pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DB if con_db else None,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


# ---------------------------------------------------------------------------
# ESQUEMA
# ---------------------------------------------------------------------------
def crear_esquema():
    # Creamos la base de datos y la tabla si no existen.
    with conectar(con_db=False) as conn, conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS {config.MYSQL_DB} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    with conectar() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pisos (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                titulo      VARCHAR(150) NOT NULL,
                ciudad      VARCHAR(50)  NOT NULL,
                barrio      VARCHAR(80)  NOT NULL,
                tipo        ENUM('piso', 'chalet') NOT NULL,
                -- Solo tiene sentido en chalets: pareado / adosado / NULL
                tipo_chalet ENUM('pareado', 'adosado') NULL,
                luminoso    BOOLEAN NOT NULL DEFAULT 0,
                ascensor    BOOLEAN NOT NULL DEFAULT 0,
                precio      INT NOT NULL,          -- en euros
                metros      INT NOT NULL,          -- metros cuadrados
                descripcion TEXT NOT NULL,
                actualizado TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP
            )
        """)


# ---------------------------------------------------------------------------
# LECTURA
# ---------------------------------------------------------------------------
def listar_pisos(ciudad=None, tipo=None, precio_max=None, metros_min=None,
                 ascensor=None, luminoso=None):
    # Construimos el WHERE dinamicamente segun los filtros que lleguen.
    # IMPORTANTE: los valores van como parametros (%s), NUNCA concatenados
    # en el string, para evitar inyeccion SQL.
    condiciones, valores = [], []
    if ciudad:
        condiciones.append("ciudad = %s"); valores.append(ciudad)
    if tipo:
        condiciones.append("tipo = %s"); valores.append(tipo)
    if precio_max:
        condiciones.append("precio <= %s"); valores.append(int(precio_max))
    if metros_min:
        condiciones.append("metros >= %s"); valores.append(int(metros_min))
    if ascensor is not None:
        condiciones.append("ascensor = %s"); valores.append(bool(ascensor))
    if luminoso is not None:
        condiciones.append("luminoso = %s"); valores.append(bool(luminoso))

    sql = "SELECT * FROM pisos"
    if condiciones:
        sql += " WHERE " + " AND ".join(condiciones)
    sql += " ORDER BY precio"

    with conectar() as conn, conn.cursor() as cur:
        cur.execute(sql, valores)
        return [_normalizar(f) for f in cur.fetchall()]


def obtener_piso(piso_id):
    with conectar() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM pisos WHERE id = %s", (piso_id,))
        fila = cur.fetchone()
        return _normalizar(fila) if fila else None


def obtener_pisos_por_ids(ids):
    # Lo usa el RAG: Qdrant nos devuelve IDs y aqui recuperamos los datos
    # frescos y oficiales de MySQL.
    if not ids:
        return []
    marcadores = ", ".join(["%s"] * len(ids))
    with conectar() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT * FROM pisos WHERE id IN ({marcadores})", ids)
        return [_normalizar(f) for f in cur.fetchall()]


# ---------------------------------------------------------------------------
# ESCRITURA (la usa el admin)
# ---------------------------------------------------------------------------
CAMPOS = ["titulo", "ciudad", "barrio", "tipo", "tipo_chalet",
          "luminoso", "ascensor", "precio", "metros", "descripcion"]


def crear_piso(datos):
    valores = [datos.get(c) for c in CAMPOS]
    sql = (f"INSERT INTO pisos ({', '.join(CAMPOS)}) "
           f"VALUES ({', '.join(['%s'] * len(CAMPOS))})")
    with conectar() as conn, conn.cursor() as cur:
        cur.execute(sql, valores)
        return cur.lastrowid  # el ID nuevo: sera tambien el ID en Qdrant


def actualizar_piso(piso_id, datos):
    asignaciones = ", ".join(f"{c} = %s" for c in CAMPOS)
    valores = [datos.get(c) for c in CAMPOS] + [piso_id]
    with conectar() as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE pisos SET {asignaciones} WHERE id = %s", valores)


def borrar_piso(piso_id):
    with conectar() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM pisos WHERE id = %s", (piso_id,))


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def _normalizar(fila):
    # MySQL devuelve los BOOLEAN como 0/1 y las fechas como datetime;
    # los pasamos a tipos que se serializan bien a JSON.
    fila["luminoso"] = bool(fila["luminoso"])
    fila["ascensor"] = bool(fila["ascensor"])
    if fila.get("actualizado"):
        fila["actualizado"] = fila["actualizado"].isoformat()
    return fila
