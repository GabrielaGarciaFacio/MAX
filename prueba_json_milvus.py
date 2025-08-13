#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import os
import sys
from urllib.parse import urljoin

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
    Vcc.Tipo_Curso AS tipo_curso_id,
    Vcc.Moneda_Precio AS nombre_moneda,
    ce.nombre AS estatus_curso,
    f.nombre AS familia_id,
    Ch.horas AS horas,
    IIF(ISNULL(Ch.link_temario,'')='', 'Temario no encontrado', Ch.link_temario) AS link_temario,
    ln.nombre AS linea_negocio_id,
    Ch.version AS version,
    e.nombre AS entrega,
    CASE 
        WHEN (Vcc.Nombre_Catalogo LIKE 'Laboratorio%' OR Vcc.Tipo_Examen = 'Certificación' OR Vcc.Curso_Tipo_Elemento = 'Examen' ) THEN Vcc.Clave
        ELSE NULL 
    END AS clave_Examen,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IS NULL THEN NULL
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificación'
        WHEN Vcc.Tipo_Examen = 'Certificación' THEN Vcc.Tipo_Examen
        WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' THEN Vcc.Nombre_Catalogo
        ELSE NULL
    END AS nombre_examen,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IN ('Examen', 'Equipo') OR Vcc.Curso_Tipo_Elemento IS NULL
        THEN Vcc.Curso_Tipo_Elemento
        ELSE NULL
    END AS tipo_elemento,
    Vcc.Base_Costo AS base_costo,
    Vcc.Pais AS pais,
    Vcc.Costo AS costo
FROM Cursos_Habilitados Ch
    LEFT JOIN vCursos_Habilitados_Costos_Integrados Vch
        ON Vch.Curso_Habilitado_Id = Ch.Id
    LEFT JOIN vCatalogos_Costos Vcc
        ON Vch.Curso_Tipo_Elemento_Id = Vcc.Curso_Tipo_Elemento_Id
        AND Vch.Curso_Elemento_Id = Vcc.Curso_Elemento_Id
        AND Vch.Curso_Elemento_Detalle_Id = Vcc.Curso_Elemento_Detalle_Id
    LEFT JOIN tecnologias t    ON Ch.tecnologia_id    = t.id
    LEFT JOIN complejidades c  ON Ch.complejidad_id   = c.id
    LEFT JOIN cursos_estatus ce ON Ch.curso_estatus_id = ce.id
    LEFT JOIN familias f       ON Ch.familia_id       = f.id  
    LEFT JOIN lineas_negocio ln ON Ch.linea_negocio_id = ln.id
    LEFT JOIN entregas e       ON Ch.entrega_id       = e.id
WHERE
    (
        (Ch.Disponible = 1
         AND (Vcc.Tipo_Curso IN ('Intensivo','Digital','Programa') OR Vcc.Tipo_Curso IS NULL)
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%'
         AND ce.nombre IN ('Liberado','Es Rentable'))
      OR
        (Ch.subcontratado = 1
         AND Ch.fin_disponibilidad >= DATEFROMPARTS(YEAR(GETDATE())-1, MONTH(GETDATE()),1)
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%'
         AND ce.nombre IN ('Liberado','Es Rentable'))
      OR
        (ce.nombre IN ('Enviado a Operaciones','Enviado a Finanzas','Enviado a Comercial','Es Rentable')
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
    );
""",
    "laboratorios": """
SELECT DISTINCT
    Ch.Clave AS clave,
    Ch.Nombre AS nombre,
    Vcc.Clave AS clave_Examen,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IN ('Examen','Equipo') OR Vcc.Curso_Tipo_Elemento IS NULL
        THEN Vcc.Curso_Tipo_Elemento
        ELSE NULL
    END AS tipo_elemento,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IS NULL THEN NULL
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificación'
        WHEN Vcc.Tipo_Examen = 'Certificación' THEN Vcc.Tipo_Examen
        WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' THEN Vcc.Nombre_Catalogo
        ELSE NULL
    END AS nombre_examen,
    Vcc.Base_Costo AS base_costo,
    Vcc.Pais AS pais,
    Vcc.Costo AS costo
FROM Cursos_Habilitados Ch
    LEFT JOIN vCursos_Habilitados_Costos_Integrados Vch
        ON Vch.Curso_Habilitado_Id = Ch.Id
    LEFT JOIN vCatalogos_Costos Vcc
        ON Vch.Curso_Tipo_Elemento_Id = Vcc.Curso_Tipo_Elemento_Id
        AND Vch.Curso_Elemento_Id         = Vcc.Curso_Elemento_Id
        AND Vch.Curso_Elemento_Detalle_Id = Vcc.Curso_Elemento_Detalle_Id
    LEFT JOIN cursos_estatus ce
        ON Ch.curso_estatus_id = ce.id
WHERE
    (
        (Ch.Disponible = 1
         AND (Vcc.Tipo_Curso IN ('Intensivo','Digital','Programa') OR Vcc.Tipo_Curso IS NULL)
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%'
         AND ce.nombre IN ('Liberado','Es Rentable'))
      OR
        (Ch.subcontratado = 1
         AND Ch.fin_disponibilidad >= DATEFROMPARTS(YEAR(GETDATE())-1, MONTH(GETDATE()),1)
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%'
         AND ce.nombre IN ('Liberado','Es Rentable'))
      OR
        (ce.nombre IN ('Enviado a Operaciones','Enviado a Finanzas','Enviado a Comercial','Es Rentable')
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%')
    )
    AND Vcc.Clave LIKE 'Lab-%';
""",
    "examenes": """
SELECT DISTINCT
    Ch.Clave AS clave,
    Ch.Nombre AS nombre,
    Vcc.Clave AS clave_Examen,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IN ('Examen','Equipo') OR Vcc.Curso_Tipo_Elemento IS NULL
        THEN Vcc.Curso_Tipo_Elemento
        ELSE NULL
    END AS tipo_elemento,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IS NULL THEN NULL
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificación'
        WHEN Vcc.Tipo_Examen = 'Certificación' THEN Vcc.Tipo_Examen
        WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' THEN Vcc.Nombre_Catalogo
        ELSE NULL
    END AS nombre_examen,
    Vcc.Base_Costo AS base_costo,
    Vcc.Pais AS pais,
    Vcc.Costo AS costo
FROM Cursos_Habilitados Ch
    LEFT JOIN vCursos_Habilitados_Costos_Integrados Vch
        ON Vch.Curso_Habilitado_Id = Ch.Id
    LEFT JOIN vCatalogos_Costos Vcc
        ON Vch.Curso_Tipo_Elemento_Id         = Vcc.Curso_Tipo_Elemento_Id
        AND Vch.Curso_Elemento_Id             = Vcc.Curso_Elemento_Id
        AND Vch.Curso_Elemento_Detalle_Id     = Vcc.Curso_Elemento_Detalle_Id
    LEFT JOIN cursos_estatus ce
        ON Ch.curso_estatus_id = ce.id
WHERE
    (
        (Ch.Disponible = 1
         AND (Vcc.Tipo_Curso IN ('Intensivo','Digital','Programa') OR Vcc.Tipo_Curso IS NULL)
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%'
         AND ce.nombre IN ('Liberado','Es Rentable'))
      OR
        (Ch.subcontratado = 1
         AND Ch.fin_disponibilidad >= DATEFROMPARTS(YEAR(GETDATE())-1, MONTH(GETDATE()),1)
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%'
         AND ce.nombre IN ('Liberado','Es Rentable'))
      OR
        (ce.nombre IN ('Enviado a Operaciones','Enviado a Finanzas','Enviado a Comercial','Es Rentable')
         AND Ch.clave NOT LIKE '%(PRIV)%'
         AND Ch.clave NOT LIKE '%(PROV)%'
         AND Ch.clave NOT LIKE '%(Servicios)%'
         AND Ch.clave NOT LIKE 'SEM%'
         AND Ch.clave NOT LIKE 'Custom%')
    )
    AND Vcc.Curso_Tipo_Elemento = 'Examen'
    AND Vcc.Tipo_Examen = 'certificación';
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
    if name not in utility.list_collections():
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        emb_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM)
        txt_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MILVUS_TEXT_MAX)
        schema = CollectionSchema(fields=[id_field, emb_field, txt_field], description=description)
        col = Collection(name=name, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 512}}
        )
        print(f"🆕 Colección creada: {name}")
    else:
        col = Collection(name)
        # Limpieza suave (mantiene índice)
        col.delete(expr="id >= 0")
        print(f"♻️  Colección vaciada: {name}")
    col.load()
    return col

def build_temarios_prefix(clave: str, nombre: str, link: str, max_prefix_len: int = 300) -> str:
    """
    Construye el prefijo 'Clave | Nombre | Link' pero acorta el link y limita
    el largo total del prefijo a max_prefix_len.
    """
    link = link or ""
    # Acortar el link si es muy largo
    if len(link) > 180:
        link_show = link[:100] + "..." + link[-60:]
    else:
        link_show = link

    prefix = f"Clave: {clave} | Nombre: {nombre} | Link: {link_show}\n\n"

    # Límite duro al tamaño del prefijo
    if len(prefix) > max_prefix_len:
        prefix = prefix[:max_prefix_len - 4] + "...\n\n"
    return prefix

def chunk_text(text: str, max_len: int = MILVUS_TEXT_MAX, prefix: str = "") -> list[str]:
    """
    Corta 'text' en fragmentos asegurando que len(prefix + fragmento) <= max_len.
    Si el prefijo es demasiado largo, se recorta para dejar al menos 50 chars disponibles.
    """
    text = (text or "").strip()
    prefix = prefix or ""

    # Asegurar que quede espacio para texto
    available = max_len - len(prefix)
    if available < 50:
        # Prefijo demasiado grande → recortarlo
        keep = max_len - 50
        if keep <= 0:
            # Caso extremo: prefijo ocupa casi todo; dejar 50 para texto
            prefix = ""
            available = 50
        else:
            prefix = prefix[:keep]
            available = 50

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= available:
            frag = prefix + remaining
            if len(frag) > max_len:
                frag = frag[:max_len]
            chunks.append(frag)
            break

        candidate = remaining[:available]
        # Cortes amigables
        cut = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"), candidate.rfind(" "))
        if cut == -1 or cut < int(available * 0.40):
            cut = available
        piece = candidate[:cut].rstrip()

        frag = prefix + piece
        if len(frag) > max_len:
            frag = frag[:max_len]
        chunks.append(frag)

        remaining = remaining[len(piece):].lstrip()

    return chunks

def record_to_text(ns: str, rec: dict) -> str:
    """
    Convierte un registro de cada JSON en texto indexable.
    - cursos/laboratorios/examenes: concatenación legible de campos.
    - temarios: sólo usa temario_texto (prefijado con metadatos).
    """
    if ns == "temarios":
        pref = f"Clave: {rec.get('clave','')} | Nombre: {rec.get('nombre','')} | Link: {rec.get('link_temario','')}\n\n"
        return pref + (rec.get("temario_texto") or "")
    elif ns == "cursos":
        fields = [
            ("Clave", "clave"), ("Nombre", "nombre"),
            ("Certificación", "certificacion"), ("Disponible", "disponible"),
            ("Sesiones", "sesiones"), ("Precio", "precio"),
            ("Subcontratado", "subcontratado"), ("Pre-requisitos", "pre_requisitos"),
            ("Tecnología", "tecnologia_id"), ("Complejidad", "complejidad_id"),
            ("Tipo de curso", "tipo_curso_id"), ("Moneda", "nombre_moneda"),
            ("Estatus", "estatus_curso"), ("Familia", "familia_id"),
            ("Horas", "horas"), ("Línea de negocio", "linea_negocio_id"),
            ("Versión", "version"), ("Entrega", "entrega"),
            ("Clave examen", "clave_Examen"), ("Nombre examen", "nombre_examen"),
            ("Tipo elemento", "tipo_elemento"), ("Base costo", "base_costo"),
            ("País", "pais"), ("Costo", "costo"),
            ("Link temario", "link_temario"),
        ]
    else:  # laboratorios / examenes
        fields = [
            ("Clave", "clave"), ("Nombre", "nombre"),
            ("Clave examen", "clave_Examen"), ("Tipo elemento", "tipo_elemento"),
            ("Nombre examen", "nombre_examen"), ("Base costo", "base_costo"),
            ("País", "pais"), ("Costo", "costo"),
        ]
    return "\n".join(f"{lbl}: {rec.get(key, '')}" for (lbl, key) in fields)

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
        batch = chunks[i:i+BATCH]
        # Seguridad extra: garantizar límite de longitud
        batch = [s if len(s) <= MILVUS_TEXT_MAX else s[:MILVUS_TEXT_MAX] for s in batch]

        vecs = embed_texts(batch)
        collection.insert([vecs, batch])
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
            clave = rec.get("clave", "")
            nombre = rec.get("nombre", "")
            link = rec.get("link_temario", "")
            cuerpo = rec.get("temario_texto") or ""

            prefix = build_temarios_prefix(clave, nombre, link)
            fragments.extend(chunk_text(cuerpo, max_len=MILVUS_TEXT_MAX, prefix=prefix))
    else:
        # cursos/laboratorios/examenes -> un texto por registro, chunqueado si excede
        for rec in data:
            base = record_to_text(ns, rec)
            fragments.extend(chunk_text(base, max_len=MILVUS_TEXT_MAX))

    # Insertar a Milvus
    print(f"➡️  Cargando {len(fragments)} fragmentos en la colección '{name}'…")
    insert_chunks(col, fragments)
    col.flush()
    print(f"✅ Colección '{name}' lista.\n")

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Exporta cursos/laboratorios/examenes/temarios a JSON y (opcional) carga en Milvus."
    )
    parser.add_argument("-n", "--namespace",
                        choices=list(QUERIES.keys()) + ["temarios", "all"],
                        default="all",
                        help="Qué exportar: 'cursos', 'laboratorios', 'examenes', 'temarios' o 'all'")
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

    def export_laboratorios():
        print("▶ Ejecutando consulta: laboratorios")
        cursor.execute(QUERIES["laboratorios"])
        rows = fetch_as_dict(cursor)
        out = os.path.join(args.output_dir, "laboratorios.json")
        write_json(rows, out)

    def export_examenes():
        print("▶ Ejecutando consulta: examenes")
        cursor.execute(QUERIES["examenes"])
        rows = fetch_as_dict(cursor)
        out = os.path.join(args.output_dir, "examenes.json")
        write_json(rows, out)

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
                "nombre": r2.get("nombre"),
                "link_temario": r2.get("link_temario"),
                "temario_texto": r2.get("temario_texto", "")
            })
        out = os.path.join(args.output_dir, "temarios.json")
        write_json(temarios, out)

    # Dispatcher de export
    to_export = ["cursos", "laboratorios", "examenes", "temarios"] if args.namespace == "all" else [args.namespace]
    for ns in to_export:
        if ns == "cursos":
            export_cursos()
        elif ns == "laboratorios":
            export_laboratorios()
        elif ns == "examenes":
            export_examenes()
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
            "laboratorios": os.path.join(args.output_dir, "laboratorios.json"),
            "examenes":     os.path.join(args.output_dir, "examenes.json"),
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
