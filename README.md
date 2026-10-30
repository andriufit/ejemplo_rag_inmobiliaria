# Inmobiliaria RAG — embeddings, búsqueda vectorial y agentes (material didáctico)

> ⚠️ **Proyecto con fines exclusivamente académicos.** Es material de clase para
> un máster de Inteligencia Artificial. El código prioriza la **legibilidad y los
> comentarios explicativos** sobre la robustez, la seguridad o el diseño visual.
> No está pensado para producción (el admin no tiene login, no hay tests, etc.).

## Qué pretendemos enseñar

Una mini inmobiliaria (Flask) con pisos de Madrid, Barcelona y Valencia para
ver **cómo se usa la búsqueda vectorial en un proyecto de software real**:
**conviven una base de datos relacional (MySQL) y una vectorial (Qdrant)**,
hay un admin para dar de alta y editar inmuebles que mantiene el RAG al día,
y un agente con memoria que ayuda a decidir qué vivienda comprar.

> **Antes de este proyecto**, en clase vimos `algoritmos.py`: la búsqueda
> vectorial hecha a mano, *sin* base de datos vectorial (métricas coseno,
> euclídea y producto escalar, búsqueda exacta por fuerza bruta y la idea de la
> búsqueda aproximada ANN/HNSW). Este repositorio es el siguiente paso.

Ideas clave que deben quedar claras:

1. **Lo que hacíamos a mano en `algoritmos.py` lo hace Qdrant por nosotros**,
   con un índice HNSW que escala a millones de vectores y la métrica del coseno.
   `comparar_embeddings.py` usa esa misma búsqueda exacta con numpy.
2. **SQL y Qdrant no compiten, se complementan**:
   - Filtros exactos (`precio <= 300000`, `ciudad = 'Madrid'`) → **MySQL**.
   - Búsqueda por significado ("algo tranquilo cerca del mar") → **Qdrant**.
3. **MySQL es la fuente de la verdad; Qdrant es un índice derivado.** El ID de
   cada punto en Qdrant es el mismo que el ID de la fila en MySQL: esa es la
   "costura" entre las dos bases de datos. Si Qdrant se pierde, se reconstruye
   desde MySQL (`POST /api/reindexar`).
4. **Cómo entran los datos nuevos en el RAG**: al crear o editar un piso en el
   admin, se guarda en MySQL y en la misma petición se recalcula su embedding y
   se hace `upsert` en Qdrant. Al borrarlo, se borra de ambos. En los
   comentarios de `app.py` se explican las limitaciones de este patrón y cómo se
   resuelve en producción (colas, reindexado periódico).
5. **Un embedding no es información literal.** Qdrant no devuelve "pisos":
   devuelve IDs y puntuaciones de similitud. La búsqueda semántica sola
   (`/api/buscar`) lo deja ver con su *score*. El agente enseña el camino
   completo hasta una respuesta útil: IDs de Qdrant → datos reales de MySQL →
   texto legible con enlace a la ficha → el LLM redacta la recomendación. El
   usuario nunca ve JSON, IDs ni puntuaciones.
6. **Un agente con memoria pregunta antes de buscar**, como lo haría un
   comercial: ciudad, tamaño y una pregunta libre ("¿qué otras cosas
   consideras importantes?"). Lo que va sabiendo lo guarda en un perfil.

## Arquitectura

```
            ┌──────────── navegador ────────────┐
            │  /  listado + filtros + chat      │
            │  /piso/<id>  ficha del inmueble   │
            │  /admin  crear / editar / borrar  │
            └────────────────┬──────────────────┘
                             │ HTTP (JSON)
                    ┌────────▼────────┐
                    │   Flask (app)   │
                    └──┬─────┬─────┬──┘
          filtros SQL  │     │     │  agente (+ memoria)
                ┌──────▼─┐ ┌─▼─────┴──────┐ ┌───────────────────────┐
                │ MySQL  │ │   Qdrant     │ │ Ollama                │
                │(verdad)│ │ (vectores)   │ │ paraphrase-multiling. │
                └────────┘ └──────────────┘ │ gemma3                │
                                            └───────────────────────┘
```

| Fichero | Responsabilidad |
|---------|-----------------|
| `config.py` | Configuración leída de `.env` |
| `db.py` | Capa relacional: esquema y CRUD con SQL parametrizado |
| `rag.py` | Capa vectorial: fila → texto → embedding → Qdrant, y búsqueda híbrida (vector + filtro por payload) |
| `agente.py` | Agente con memoria: **entrevista** (3 preguntas) → **búsqueda híbrida** en Qdrant → **respuesta** en texto con enlaces |
| `app.py` | API REST y sincronización MySQL → Qdrant |
| `seed.py` | Carga 11 viviendas de ejemplo (5 Madrid, 3 Barcelona, 3 Valencia) |
| `comparar_embeddings.py` | Mini evaluación: compara varios modelos de embedding con preguntas en español |
| `templates/` | Front mínimo (HTML + JS sin frameworks): listado, ficha del inmueble y admin |

### Cómo funciona el agente

```
 "Hola, busco algo en Valencia de unos 70 m2"
        │
        ▼
 1. ENTREVISTA ── ¿falta ciudad, tamaño o "qué más te importa"? ── sí ──► pregunta
        │ no                                                         (texto fijo,
        ▼                                                             sin LLM)
 2. BÚSQUEDA HÍBRIDA en Qdrant
      embedding("cerca de la playa, tranquilo...")   ← significado
      + filtro ciudad = Valencia, metros ≥ 70         ← datos exactos
        │  devuelve: [(10, 0.63), (11, 0.47), (9, 0.45)]  ← ¡solo IDs y scores!
        ▼
 3. DE IDS A INFORMACIÓN
      IDs → MySQL → "Casa de pueblo en el Cabanyal: 75 m2, 210.000 €...
                     Enlace: http://localhost:8000/piso/10"
        │
        ▼
 4. GEMMA3 redacta la recomendación (con el historial = memoria)
```

- **Memoria**: cada conversación (un ID que genera el navegador) guarda en el
  servidor su **historial** de mensajes y un **perfil** estructurado
  (ciudad, metros, preferencias). El chat muestra lo que el agente recuerda.
  Está en RAM a propósito, para que sea sencillo: se pierde al reiniciar
  Flask. En producción iría en Redis o en una tabla.
- **Ciudad y metros se extraen con código, no con el LLM**: son datos cerrados
  y una regla sencilla es más fiable que un modelo pequeño, que a veces se
  inventa ciudades. Usa un LLM solo donde aporte algo.
- **Lo interno queda en el servidor**: la consulta, los filtros y los
  IDs/scores de Qdrant se imprimen en la consola de Flask (`[agente] ...`),
  y es ahí donde hay que mirarlos en clase.

## Cómo levantarlo

### Requisito: Ollama con los modelos

En las dos opciones, [Ollama](https://ollama.com) corre en tu máquina con estos modelos:

```bash
ollama pull paraphrase-multilingual   # embeddings (768 dimensiones, multilingüe)
ollama pull gemma3                    # LLM del agente
```

### Opción A: Docker (la más rápida para una demo)

Solo necesita Docker y Ollama. Levanta MySQL (MariaDB), Qdrant y la app; los
datos de ejemplo se cargan solos la primera vez.

```bash
docker compose up --build
```

| Qué | URL |
|-----|-----|
| Web (listado, filtros, agente) | http://localhost:8080 |
| Ficha de un inmueble | http://localhost:8080/piso/1 |
| Admin (alta, edición, borrado) | http://localhost:8080/admin |
| Panel de Qdrant | http://localhost:6343/dashboard |

- Ver lo que hace el agente por dentro: `docker compose logs -f app`
- Empezar de cero (borra los datos): `docker compose down -v`

### Opción B: en local (Python + XAMPP + Qdrant)

Requiere Python 3.10+.

1. Arranca **MySQL desde XAMPP** (panel de control → *Start* en MySQL).
2. Arranca **Qdrant** en el puerto 6333:
   ```bash
   docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
   ```
3. Instala dependencias y configura:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env      # revisa MYSQL_PORT (3306 por defecto en XAMPP)
   ```
4. Carga los datos de ejemplo (crea la BBDD, la tabla y el índice en Qdrant):
   ```bash
   python seed.py
   ```
5. Arranca la app:
   ```bash
   python app.py
   ```

| Qué | URL |
|-----|-----|
| Web | http://localhost:8000 |
| Admin | http://localhost:8000/admin |
| phpMyAdmin (XAMPP) | http://localhost/phpmyadmin |
| Panel de Qdrant | http://localhost:6333/dashboard |

> **Problemas típicos en Windows**
> - *"Intento de acceso a un socket no permitido"*: el puerto 5000 de Flask
>   está reservado por Windows; por eso la app usa el 8000 (`PORT` en `.env`).
> - *MySQL de XAMPP no arranca*: otro MySQL (p. ej. el servicio `MySQL80`)
>   ocupa el 3306. Para ese servicio o arranca XAMPP en otro puerto y ponlo en
>   `MYSQL_PORT`.
> - *El agente responde "llama-server process has terminated ... CUDA error"*:
>   Ollama se ha quedado sin memoria de GPU (portátiles con 6 GB). Descarga los
>   modelos que no uses con `ollama stop <modelo>` (ver `ollama ps`) y reintenta.

## API

| Método | Ruta | Qué hace | BBDD |
|--------|------|----------|------|
| GET | `/api/pisos?ciudad=&tipo=&precio_max=&metros_min=&ascensor=1&luminoso=0` | Listado con filtros | MySQL |
| GET | `/api/pisos/<id>` | Un piso | MySQL |
| POST | `/api/pisos` | Crear piso | MySQL + Qdrant |
| PUT | `/api/pisos/<id>` | Editar piso | MySQL + Qdrant |
| DELETE | `/api/pisos/<id>` | Borrar piso | MySQL + Qdrant |
| GET | `/api/buscar?q=...` | Búsqueda semántica pura (devuelve el *score*) | Qdrant |
| POST | `/api/agente` | Conversar con el agente (`{"conversacion_id": "...", "mensaje": "..."}`) | Ollama + Qdrant + MySQL |
| GET | `/piso/<id>` | Ficha HTML de un inmueble (a donde enlaza el agente) | MySQL |
| POST | `/api/reindexar` | Reconstruir Qdrant desde MySQL | MySQL → Qdrant |
