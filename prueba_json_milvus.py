#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import os
import sys
from urllib.parse import urljoin
import re
import pyodbc
import requests
from PyPDF2 import PdfReader

# =========================
# Config SQL (no cambia)
# =========================
CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;"
    "DATABASE=netec_prod;"
    "UID=netec_read;"
    "PWD=R3ad25**SC3.2025-;"
    "TrustServerCertificate=yes;"
)

# =========================
# Consultas
# =========================
QUERIES = {
    "cursos": """
SELECT DISTINCT  
        Ch.Clave AS clave,
        Ch.Nombre AS nombre,
        Ch.Certificacion AS certificacion,
        IIF(Ch.Disponible = 1, 'Si', 'No') AS disponible,
        Ch.Sesiones AS sesiones,
        Ch.pecio_lista AS precio,
        IIF(Ch.subcontratado = 1, 'Si', 'No') AS subcontratado,
        Ch.pre_requisitos AS pre_requisitos,
        t.nombre AS tecnologia_id, 
        c.nombre AS complejidad_id,
        -- Tomar el valor no nulo entre examen y laboratorio
        COALESCE(
            MAX(CASE WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN Vcc.Tipo_Curso END),
            MAX(CASE WHEN Vcc.Curso_Tipo_Elemento = 'Equipo' THEN Vcc.Tipo_Curso END)
        ) AS tipo_curso_id,
        -- Tomar el valor no nulo entre examen y laboratorio
        COALESCE(
            MAX(CASE WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN Vcc.Moneda_Precio END),
            MAX(CASE WHEN Vcc.Curso_Tipo_Elemento = 'Equipo' THEN Vcc.Moneda_Precio END)
        ) AS nombre_moneda,
        ce.nombre AS estatus_curso,
        f.nombre AS familia_id,
        Ch.horas AS horas,
        IIF(ISNULL(Ch.link_temario,'')='', 'Temario no encontrado', Ch.link_temario) AS link_temario,
        ln.nombre AS linea_negocio_id,
        Ch.version AS version,
        e.nombre AS entrega,
        p.nombre AS pais_curso,

        -- CAMPOS CONSOLIDADOS PARA EXAMEN
        MAX(CASE 
            WHEN Vcc.Curso_Tipo_Elemento = 'Examen' OR Vcc.Tipo_Examen = 'Certificaci n'
            THEN Vcc.Clave
            ELSE NULL 
        END) AS clave_examen,

        MAX(CASE
            WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificaci n'
            WHEN Vcc.Tipo_Examen = 'Certificaci n' THEN Vcc.Tipo_Examen
            ELSE NULL
        END) AS nombre_examen,

        MAX(CASE
            WHEN Vcc.Curso_Tipo_Elemento = 'Examen' OR Vcc.Tipo_Examen = 'Certificaci n'
            THEN Vcc.Base_Costo
            ELSE NULL
        END) AS base_costo_examen,

        MAX(CASE
            WHEN Vcc.Curso_Tipo_Elemento = 'Examen' OR Vcc.Tipo_Examen = 'Certificaci n'
            THEN Vcc.Costo
            ELSE NULL
        END) AS costo_examen,

        -- CAMPOS CONSOLIDADOS PARA LABORATORIO
        MAX(CASE 
            WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' AND Vcc.Curso_Tipo_Elemento = 'Equipo'
            THEN Vcc.Clave
            ELSE NULL 
        END) AS clave_laboratorio,

        MAX(CASE
            WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' AND Vcc.Curso_Tipo_Elemento = 'Equipo'
            THEN Vcc.Nombre_Catalogo
            ELSE NULL
        END) AS nombre_laboratorio,

        MAX(CASE
            WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' AND Vcc.Curso_Tipo_Elemento = 'Equipo'
            THEN Vcc.Base_Costo
            ELSE NULL
        END) AS base_costo_laboratorio,

        MAX(CASE
            WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' AND Vcc.Curso_Tipo_Elemento = 'Equipo'
            THEN Vcc.Costo
            ELSE NULL
        END) AS costo_laboratorio,

        -- PA S (el mismo para ambos elementos)
        Vcc.Pais AS pais

    FROM Cursos_Habilitados Ch
        LEFT OUTER JOIN vCursos_Habilitados_Costos_Integrados Vch
            ON Vch.Curso_Habilitado_Id = Ch.Id
        LEFT OUTER JOIN vCatalogos_Costos Vcc
            ON Vch.Curso_Tipo_Elemento_Id = Vcc.Curso_Tipo_Elemento_Id
            AND Vch.Curso_Elemento_Id = Vcc.Curso_Elemento_Id
            AND Vch.Curso_Elemento_Detalle_Id = Vcc.Curso_Elemento_Detalle_Id
        LEFT OUTER JOIN tecnologias t
            ON Ch.tecnologia_id = t.id
        LEFT OUTER JOIN complejidades c
            ON Ch.complejidad_id = c.id
        LEFT OUTER JOIN cursos_estatus ce
            ON Ch.curso_estatus_id = ce.id
        LEFT OUTER JOIN familias f
            ON Ch.familia_id = f.id  
        LEFT OUTER JOIN lineas_negocio ln
            ON Ch.linea_negocio_id = ln.id
        LEFT OUTER JOIN entregas e
            ON Ch.entrega_id = e.id
        LEFT OUTER JOIN paises p
            ON Ch.pais_id = p.id
    WHERE
        (
            (Ch.Disponible = 1
            AND (Vcc.Tipo_Curso IN ('Intensivo', 'Digital', 'Programa') OR Vcc.Tipo_Curso IS NULL)
            AND Ch.clave NOT LIKE '%(PRIV)%'
            AND Ch.clave NOT LIKE '%(PROV)%'
            AND Ch.clave NOT LIKE '%(Servicios)%'
            AND Ch.clave NOT LIKE 'SEM%'
            AND Ch.clave NOT LIKE 'Custom%'
            AND ce.nombre IN ('Liberado', 'Es Rentable'))

            OR

            (Ch.subcontratado = 1
            AND Ch.fin_disponibilidad >= DATEFROMPARTS(YEAR(GETDATE())-1, MONTH(GETDATE()),1)
            AND Ch.clave NOT LIKE '%(PRIV)%'
            AND Ch.clave NOT LIKE '%(PROV)%'
            AND Ch.clave NOT LIKE '%(Servicios)%'
            AND Ch.clave NOT LIKE 'SEM%'
            AND Ch.clave NOT LIKE 'Custom%'
            AND ce.nombre IN ('Liberado', 'Es Rentable'))

            OR

            (ce.nombre IN ('Enviado a Operaciones', 'Enviado a Finanzas', 'Enviado a Comercial', 'Es Rentable')
            AND Ch.clave NOT LIKE '%(PRIV)%'
            AND Ch.clave NOT LIKE '%(PROV)%'
            AND Ch.clave NOT LIKE '%(Servicios)%'
            AND Ch.clave NOT LIKE 'SEM%'
            AND Ch.clave NOT LIKE 'Custom%')
        )
        AND (
            Vcc.Curso_Tipo_Elemento IS NULL
            OR Vcc.Curso_Tipo_Elemento = 'Examen'
            OR (Vcc.Curso_Tipo_Elemento = 'Equipo' AND Vcc.Nombre_Catalogo LIKE 'Labo%')
        )
        AND Ch.clave IS NOT NULL 
        AND Ch.clave != ''
        AND LTRIM(RTRIM(Ch.clave)) != ''
        AND Ch.clave LIKE '%AWS%'

    GROUP BY 
        Ch.Clave,
        Ch.Nombre,
        Ch.Certificacion,
        Ch.Disponible,
        Ch.Sesiones,
        Ch.pecio_lista,
        Ch.subcontratado,
        Ch.pre_requisitos,
        t.nombre,
        c.nombre,
        ce.nombre,
        f.nombre,
        Ch.horas,
        Ch.link_temario,
        ln.nombre,
        Ch.version,
        e.nombre,
        p.nombre,
        Vcc.Pais

    -- Filtrar  nicamente las filas completamente vac as (sin labs, sin ex menes Y sin pa s)
    HAVING NOT (
        MAX(CASE WHEN Vcc.Curso_Tipo_Elemento = 'Examen' OR Vcc.Tipo_Examen = 'Certificaci n' THEN Vcc.Clave ELSE NULL END) IS NULL
        AND MAX(CASE WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' AND Vcc.Curso_Tipo_Elemento = 'Equipo' THEN Vcc.Clave ELSE NULL END) IS NULL
        AND Vcc.Pais IS NULL
    )

    ORDER BY clave, pais
 """
}

# =========================
# Utilidades extracción/normalización
# =========================
def fetch_as_dict(cursor):
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

def extract_pdf_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as e:
        return f"[ERROR extrayendo PDF] {e}"

def normalize_link_only(record: dict, base_url="https://sce.netec.com/"):
    lt = record.get("link_temario")
    if lt and lt != "Temario no encontrado":
        full = lt if lt.startswith("http") else urljoin(base_url, lt)
        record["link_temario"] = requests.utils.requote_uri(full)
    return record

def normalize_link_and_extract(record: dict, base_url="https://sce.netec.com/"):
    lt = record.get("link_temario")
    if lt and lt != "Temario no encontrado":
        full = lt if lt.startswith("http") else urljoin(base_url, lt)
        full_escaped = requests.utils.requote_uri(full)
        record["link_temario"]  = full_escaped
        record["temario_texto"] = extract_pdf_text(full_escaped)
    return record

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def write_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Escrito: {path}  (items: {len(data) if isinstance(data, list) else 'n/a'})")

# ---- Dedupe para TEMARIOS (una extracción por clave) ----
def dedupe_temarios_por_clave(rows):
    """
    Devuelve items únicos por 'clave', priorizando el primer link_temario válido.
    Estructura de salida: {'clave','nombre','link_temario'}
    """
    por_clave = {}
    for r in rows:
        clave = r.get("clave")
        if not clave:
            continue
        lt = r.get("link_temario")
        actual = por_clave.get(clave)
        if actual is None:
            por_clave[clave] = {"clave": clave, "nombre": r.get("nombre"), "link_temario": lt}
        else:
            actual_ok = actual.get("link_temario") and actual["link_temario"] != "Temario no encontrado"
            nuevo_ok  = lt and lt != "Temario no encontrado"
            if (not actual_ok) and nuevo_ok:
                actual["link_temario"] = lt
            if not actual.get("nombre") and r.get("nombre"):
                actual["nombre"] = r.get("nombre")
    return list(por_clave.values())

# =========================
# Carga a Milvus (opcional)
# =========================
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536
MILVUS_TEXT_MAX = 2500  # tamaño máximo del VARCHAR y chunking

def connect_milvus(host: str, port: int):
    from pymilvus import connections
    connections.connect(alias="default", host=host, port=port)

def setup_collection(name: str, description: str = ""):
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

    def create_new():
        id_field   = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        emb_field  = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM)
        txt_field  = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MILVUS_TEXT_MAX)
        pais_field      = FieldSchema(name="pais", dtype=DataType.VARCHAR, max_length=64)
        paiscurso_field = FieldSchema(name="pais_curso", dtype=DataType.VARCHAR, max_length=64)

        schema = CollectionSchema(
            fields=[id_field, emb_field, txt_field, pais_field, paiscurso_field],
            description=description
        )
        col = Collection(name=name, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 512}}
        )
        print(f"🆕 Colección creada: {name}")
        return col

    if name in utility.list_collections():
        col = Collection(name)
        existing = {f.name for f in col.schema.fields}
        needed = {"id", "embedding", "text", "pais", "pais_curso"}  # esquema objetivo
        if not needed.issubset(existing):
            col.release()
            utility.drop_collection(name)
            print(f"🗑️  Colección '{name}' eliminada por cambio de esquema: {existing} -> {needed}")
            col = create_new()
        else:
            col.delete(expr="id >= 0")
            print(f"♻️  Colección vaciada: {name}")
    else:
        col = create_new()

    col.load()
    return col

def build_temarios_prefix(clave: str, nombre: str, link: str, max_prefix_len: int = 300) -> str:
    """
    Prefijo SOLO con clave + nombre (sin link), con salto de línea limpio y límite.
    """
    nombre = nombre or ""
    prefix = f"Clave: {clave}\nNombre: {nombre}\n\n"
    if len(prefix) > max_prefix_len:
        prefix = prefix[:max_prefix_len - 4] + "...\n\n"
    return prefix

def chunk_text(text: str, max_len: int, prefix: str = "") -> list[str]:
    """
    Divide 'text' en trozos que, al anteponer 'prefix', NO superen 'max_len' BYTES UTF-8.
    - Conserva TODO el contenido (no trunca).
    - Intenta cortar en límites de palabra; si una palabra excede, la divide sin perder bytes.
    """
    text = text or ""
    prefix = prefix or ""
    max_bytes = max_len

    pref_b = prefix.encode("utf-8")
    if len(pref_b) >= max_bytes:
        # Prefijo ya llena el cupo → devuelve solo prefijo recortado a bytes (no hay cuerpo posible)
        safe_pref = pref_b[:max_bytes].decode("utf-8", errors="ignore")
        return [safe_pref]

    body_limit = max_bytes - len(pref_b)
    # Tokeniza preservando espacios (palabras o secuencias de espacio)
    tokens = re.findall(r"\S+|\s+", text)

    chunks = []
    cur_parts = []
    cur_bytes = 0

    def flush():
        if cur_parts:
            body = "".join(cur_parts)
            chunks.append(prefix + body)

    for tok in tokens:
        tok_b = tok.encode("utf-8")
        blen = len(tok_b)

        if blen <= (body_limit - cur_bytes):
            # cabe entero en el chunk actual
            cur_parts.append(tok)
            cur_bytes += blen
            continue

        if cur_bytes > 0:
            # cierra chunk actual y empieza uno nuevo
            flush()
            cur_parts, cur_bytes = [], 0

        # ahora tenemos chunk vacío; si el token aún no cabe, partirlo por bytes
        start = 0
        while start < blen:
            remaining = body_limit - cur_bytes
            piece_b = tok_b[start:start + remaining]
            piece = piece_b.decode("utf-8", errors="ignore")
            # recalcula por si el decode recortó al borde de un multibyte
            piece_b = piece.encode("utf-8")
            start += len(piece_b)
            cur_parts.append(piece)
            cur_bytes += len(piece_b)

            if cur_bytes == body_limit:
                flush()
                cur_parts, cur_bytes = [], 0

    # último flush
    flush()

    # Caso borde: texto vacío → al menos devolver prefijo (si existe)
    if not chunks and prefix:
        return [prefix]
    return chunks

#Función para detectar idioma basado en la clave
def detect_language_from_key(clave: str) -> str:
    """
    Detecta el idioma del curso basado en la clave.
    Si contiene 'ESP' es español, si no, inglés.
    """
    if isinstance(clave, str) and 'ESP' in clave.upper():
        return "Español"
    return "Inglés"

def record_to_text(ns: str, rec: dict) -> str:
    """
    Convierte un registro en texto indexable.
    - cursos: concatenación legible de campos.
    - temarios: usa temario_texto prefijado con metadatos (clave y nombre).
    """

    if ns == "temarios":
        pref = f"Clave: {rec.get('clave','')} | Nombre: {rec.get('nombre','')}\n\n"
        return pref + (rec.get("temario_texto") or "")
    else:
        clave = rec.get('clave', '')
        idioma = detect_language_from_key(clave)

        # 🔧 Campos EXACTOS que devuelve tu SELECT
        fields = [
            ("Clave", "clave"),
            ("Nombre", "nombre"),
            ("Certificación", "certificacion"),
            ("Disponible", "disponible"),
            ("Sesiones", "sesiones"),
            ("Precio", "precio"),
            ("Subcontratado", "subcontratado"),
            ("Pre-requisitos", "pre_requisitos"),
            ("Tecnología", "tecnologia_id"),
            ("Complejidad", "complejidad_id"),
            ("Tipo de curso", "tipo_curso_id"),
            ("Moneda", "nombre_moneda"),
            ("Estatus", "estatus_curso"),
            ("Familia", "familia_id"),
            ("Horas", "horas"),
            ("Línea de negocio", "linea_negocio_id"),
            ("Versión", "version"),
            ("Entrega", "entrega"),

            # País en catálogo y país del elemento de costos
            ("País del curso", "pais_curso"),
            ("País (elemento)", "pais"),

            # Consolidados de EXAMEN
            ("Clave examen", "clave_examen"),
            ("Nombre examen", "nombre_examen"),
            ("Base costo examen", "base_costo_examen"),
            ("Costo examen", "costo_examen"),

            # Consolidados de LABORATORIO
            ("Clave laboratorio", "clave_laboratorio"),
            ("Nombre laboratorio", "nombre_laboratorio"),
            ("Base costo laboratorio", "base_costo_laboratorio"),
            ("Costo laboratorio", "costo_laboratorio"),

            ("Link temario", "link_temario"),
        ]

        lines = [f"{lbl}: {rec.get(key, '')}" for (lbl, key) in fields]
        lines.append(f"Idioma: {idioma}")
        return "\n".join(lines)

def embed_texts(texts):
    # Embedding en batch con OpenAIEmbeddings (LangChain)
    from langchain_community.embeddings import OpenAIEmbeddings
    emb = OpenAIEmbeddings(model=EMB_MODEL)
    # OpenAIEmbeddings ya hace batching interno; si quieres controlar el tamaño, se puede mapear a mano.
    vectors = emb.embed_documents(texts)
    return vectors

def insert_chunks(collection, chunks):
    from math import ceil
    BATCH = 64
    total = len(chunks)
    for i in range(0, total, BATCH):
        part = chunks[i:i+BATCH]
        texts       = [c["text"] for c in part]
        paises      = [c.get("pais", "NA") for c in part]
        pais_cursos = [c.get("pais_curso", "NA") for c in part]

        # Validación por BYTES (sin truncar)
        for idx_chk, s in enumerate(texts):
            blen = len(s.encode("utf-8"))
            if blen > MILVUS_TEXT_MAX:
                raise ValueError(f"Chunk excede {MILVUS_TEXT_MAX} bytes (got {blen}) en batch {i} idx {idx_chk}")

        vecs = embed_texts(texts)
        collection.insert([vecs, texts, paises, pais_cursos])

        print(f"\r   > Insertados {min(i+BATCH, total)}/{total}", end="")
    print(" ✔")

def load_json_to_milvus(ns: str, path: str, collection_prefix: str):
    """
    Carga un JSON a su colección:
    - netec_cursos, netec_laboratorios, netec_examenes, netec_temarios
    """
    from pymilvus import Collection
    name = f"{collection_prefix}{ns}"
    col_desc = f"Índice para {ns}"
    col = setup_collection(name, description=col_desc)

    # Leer JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Armar lista de fragmentos
    fragments = []
    if ns == "temarios":
        for rec in data:
            clave  = rec.get("clave", "")
            nombre = rec.get("nombre", "")
            link   = rec.get("link_temario", "")
            cuerpo = rec.get("temario_texto") or ""

            prefix = build_temarios_prefix(clave, nombre, link)
            for ch in chunk_text(cuerpo, max_len=MILVUS_TEXT_MAX, prefix=prefix):
                fragments.append({
                    "text": ch,
                    "pais": "NA",
                    "pais_curso": "NA",
                })
    else:
        # cursos -> un texto por registro (chunked si excede)
        for rec in data:
            base = record_to_text(ns, rec)
            pais_elem  = rec.get("pais") or "NA"
            pais_curso = rec.get("pais_curso") or "NA"
            for ch in chunk_text(base, max_len=MILVUS_TEXT_MAX):
                fragments.append({
                    "text": ch,
                    "pais": pais_elem,
                    "pais_curso": pais_curso,
                })

    # --- DEBUG/EXPORT: guardar cómo quedó el chunking ---
    out_chunks = os.path.join(os.path.dirname(path), f"{ns}_chunks.json")
    try:
        with open(out_chunks, "w", encoding="utf-8") as fch:
            json.dump(fragments, fch, ensure_ascii=False, indent=2)
        print(f"🧩 Chunks escritos en: {out_chunks}  (fragments: {len(fragments)})")
    except Exception as e:
        print(f"[WARN] No se pudo escribir {ns}_chunks.json: {e}")

    if fragments:
        mx = max(len(f["text"].encode("utf-8")) for f in fragments)
        print(f"🔎 Máx. longitud (bytes) en '{ns}': {mx} (límite {MILVUS_TEXT_MAX})")

    # Insertar a Milvus (sin abortar todo si falla)
    try:
        print(f"➡️  Cargando {len(fragments)} fragmentos en la colección '{name}'…")
        insert_chunks(col, fragments)
        col.flush()
        print(f"✅ Colección '{name}' lista.\n")
    except Exception as e:
        print(f"[ERROR] Insertando en '{name}': {e}. Se continúa con el siguiente namespace.\n")

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Exporta cursos y temarios a JSON y (opcional) carga en Milvus."
    )
    parser.add_argument("-n", "--namespace",
                        choices=list(QUERIES.keys()) + ["temarios", "all"],
                        default="all",
                        help="Qué exportar: 'cursos', 'temarios' o 'all'")
    parser.add_argument("--output-dir", default=".", help="Directorio de salida (por defecto: .)")

    # Flags Milvus
    parser.add_argument("--milvus-load", action="store_true",
                        help="Si se especifica, carga a Milvus los JSON generados (o existentes en --output-dir).")
    parser.add_argument("--milvus-host", default="localhost", help="Host de Milvus (por defecto: localhost)")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Puerto de Milvus (por defecto: 19530)")
    parser.add_argument("--collection-prefix", default="",
                        help="Prefijo para colecciones (por defecto: "" ")

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    # Conectar SQL
    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    cursos_cache = None  # 'cursos' normalizados (sin extracción)

    # ---------- Exportadores ----------
    def export_cursos():
        nonlocal cursos_cache
        print("▶ Ejecutando consulta: cursos")
        cursor.execute(QUERIES["cursos"])
        rows = fetch_as_dict(cursor)
        total = len(rows)
        processed = []
        for idx, rec in enumerate(rows, start=1):
            pct = (idx * 100.0) / max(total, 1)
            print(f"[{idx}/{total}] Normalizando links… {pct:.1f}%", flush=True)
            processed.append(normalize_link_only(rec))
        cursos_cache = processed
        out = os.path.join(args.output_dir, "cursos.json")
        write_json(processed, out)

    def export_temarios():
        """
        Derivado de 'cursos' AGRUPANDO por 'clave' (dedupe) y extrayendo PDF una única vez por clave.
        """
        nonlocal cursos_cache
        if cursos_cache is None:
            print("▶ Cargando cursos para derivar temarios…")
            cursor.execute(QUERIES["cursos"])
            rows = fetch_as_dict(cursor)
            cursos_cache = [normalize_link_only(r) for r in rows]

        unicos = dedupe_temarios_por_clave(cursos_cache)
        candidatos = [u for u in unicos if u.get("link_temario") and u["link_temario"] != "Temario no encontrado"]

        total = len(candidatos)
        temarios = []
        for idx, item in enumerate(candidatos, start=1):
            print(f"[{idx}/{total}] Extrayendo PDF de temario (clave={item.get('clave')})…", flush=True)
            r2 = dict(item)
            normalize_link_and_extract(r2)
            temarios.append({
                "clave": r2.get("clave"),
                "nombre": r2.get("nombre", ""),   
                "temario_texto": r2.get("temario_texto", "")
            })
        out = os.path.join(args.output_dir, "temarios.json")
        write_json(temarios, out)

    # Dispatcher de export
    to_export = ["cursos", "temarios"] if args.namespace == "all" else [args.namespace]
    for ns in to_export:
        if ns == "cursos":
            export_cursos()
        elif ns == "temarios":
            export_temarios()

    # ---------- Carga a Milvus (opcional) ----------
    if args.milvus_load:
        try:
            connect_milvus(args.milvus_host, args.milvus_port)
        except Exception as e:
            print(f"[ERROR] No se pudo conectar a Milvus: {e}")
            sys.exit(3)

        # Mapa ns -> archivo esperado
        ns_files = {
            "cursos":       os.path.join(args.output_dir, "cursos.json"),
            "temarios":     os.path.join(args.output_dir, "temarios.json"),
        }
        for ns in to_export:
            path = ns_files.get(ns)
            if not path or not os.path.isfile(path):
                print(f"[WARN] No se encontró JSON para '{ns}' en {args.output_dir}. Omitiendo.")
                continue
            load_json_to_milvus(ns, path, collection_prefix=args.collection_prefix)

if __name__ == "__main__":
    main()
