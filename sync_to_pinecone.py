import os
import hashlib
import pyodbc
import openai
from pinecone import Pinecone
from datetime import datetime
from models import conf
import unicodedata
import re

import requests

PINECONE_HOST="https://rag-wib1770.svc.eastus-azure.pinecone.io"

# Extraes las claves
api_key_open = conf.settings.openai_api_key.get_secret_value()
pinecone_key = conf.settings.pinecone_api_key.get_secret_value()

# Configuras OpenAI
openai.api_key = api_key_open

# Creas instancia de Pinecone
pc = Pinecone(api_key=pinecone_key)

# Accedes al índice
index = pc.Index("rag")


# Conexión a SQL Server
def conectar_sql():
    return pyodbc.connect("DRIVER={ODBC Driver 18 for SQL Server};SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;DATABASE=netec_prod;UID=netec_read;PWD=R3ad25**SC3.2025-;TrustServerCertificate=yes;")

# Generación de embeddings
def embed(texto):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding

# Hash para control de cambios
def generar_hash(texto):
    return hashlib.md5(texto.encode('utf-8')).hexdigest()

#Limpieza de metadata para Pinecone
def limpiar_metadata(metadata: dict) -> dict:
    limpio = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            # Si es string vacío o None, convertir a string vacío
            if v is None or (isinstance(v, str) and v.strip() == ''):
                limpio[k] = ""
            else:
                limpio[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            limpio[k] = v
        elif v is None:
            limpio[k] = ""  # Convertir null a string vacío
        else:
            limpio[k] = str(v) if v is not None else ""  # como último recurso, evitar null
    return limpio

def limpiar_id_para_pinecone(texto):
    """Limpia el texto para que sea un ID válido en Pinecone"""
    if not texto:
        return ""
    # Quitar acentos/tildes
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    # Solo eliminar caracteres que realmente causan problemas en Pinecone
    texto = re.sub(r"[^\w\s\-\(\)]", "", texto)
    # Convertir espacios a guiones
    texto = re.sub(r"\s+", "-", texto)
    # Limpiar guiones múltiples
    texto = re.sub(r"-+", "-", texto)
    # Quitar guiones al inicio y final
    texto = texto.strip("-")
    return texto

def generar_id_unico_simple(row_data):
    """
    NUEVA función que genera un ID único basado en ROW_NUMBER para evitar cualquier duplicado.
    Combina información clave + un hash único del registro completo.
    """
    try:
        # Información clave para el ID
        clave = str(row_data.get('clave', '')).strip()
        pais = str(row_data.get('pais', 'Sin_Pais')).strip()
        
        # Crear un hash único basado en TODO el contenido del registro
        # Esto garantiza que cada registro único tenga un ID único
        contenido_completo = ""
        for key in sorted(row_data.keys()):  # Ordenar claves para consistencia
            valor = str(row_data.get(key, ''))
            contenido_completo += f"{key}:{valor}|"
        
        # Hash único de todo el registro
        hash_completo = hashlib.md5(contenido_completo.encode('utf-8')).hexdigest()
        
        # Limpiar clave para ID
        clave_limpia = limpiar_id_para_pinecone(clave)
        if len(clave_limpia) > 20:  # Limitar longitud
            clave_limpia = clave_limpia[:20]
        
        # Limpiar país para ID
        pais_limpio = limpiar_id_para_pinecone(pais) if pais != 'Sin_Pais' else ""
        
        # ID final: clave + país (si existe) + hash único
        if pais_limpio:
            id_final = f"{clave_limpia}_{pais_limpio}_{hash_completo[:16]}"
        else:
            id_final = f"{clave_limpia}_{hash_completo[:16]}"
            
        # Asegurar que el ID no esté vacío
        if not id_final or id_final.startswith("_"):
            id_final = f"curso_{hash_completo[:16]}"
            
        return id_final
        
    except Exception as e:
        print(f"ERROR generando ID único: {e}")
        # Fallback: usar solo el hash completo
        contenido_fallback = str(row_data)
        hash_fallback = hashlib.md5(contenido_fallback.encode('utf-8')).hexdigest()
        return f"fallback_{hash_fallback[:16]}"

# ---------- SYNC CURSOS COMPLETAMENTE REVISADO ----------
def sync_cursos():
    print(f"[{datetime.now()}] Iniciando sincronización de cursos...")
    conn = conectar_sql()
    cursor = conn.cursor()

    # Tu query EXACTO sin modificaciones
    query_cursos = """
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

        -- Solo mantener clave_Examen si nombre_examen es 'certificación' o empieza con "LAB", de lo contrario NULL
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
    WHERE
        (
            -- Tus condiciones generales (sin tocar)
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
        -- Filtro básico para claves válidas
        AND Ch.clave IS NOT NULL 
        AND Ch.clave != ''
        AND LTRIM(RTRIM(Ch.clave)) != ''
    """
    
    print(f"[{datetime.now()}] Ejecutando consulta SQL...")
    cursor.execute(query_cursos)
    columns = [col[0] for col in cursor.description]
    registros = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    print(f"[{datetime.now()}] Encontrados {len(registros)} registros ÚNICOS en la base de datos")
    
    # Validación básica sin deduplicación adicional (ya viene con DISTINCT)
    registros_validos = []
    registros_invalidos = 0
    
    for row in registros:
        clave = row.get('clave')
        if not clave or str(clave).strip() == "" or str(clave).strip().lower() == "none":
            registros_invalidos += 1
            continue
        
        # Limpiar valores None en país
        if not row.get('pais'):
            row['pais'] = 'Sin_Pais'
            
        registros_validos.append(row)
    
    print(f"[{datetime.now()}] Registros válidos: {len(registros_validos)}")
    print(f"[{datetime.now()}] Registros inválidos (clave vacía): {registros_invalidos}")
    
    # Procesar TODOS los registros válidos
    ids_validos_bd = set()
    vectores_para_upsert = []
    contador_actualizados = 0
    contador_sin_cambios = 0
    contador_errores = 0
    contador_ids_duplicados = 0
    
    # Diccionario para detectar IDs duplicados durante la generación
    ids_generados = {}
    
    # Procesar registros en lotes
    batch_size = 100
    total_registros = len(registros_validos)
    
    for i in range(0, total_registros, batch_size):
        batch = registros_validos[i:i + batch_size]
        print(f"[{datetime.now()}] Procesando lote {i//batch_size + 1}/{(total_registros-1)//batch_size + 1} ({len(batch)} registros)")
        
        # Generar IDs únicos para el lote
        batch_data = {}
        for row in batch:
            try:
                # Generar ID único usando la nueva función
                id_unico = generar_id_unico_simple(row)
                
                # VALIDAR que el ID no esté duplicado
                if id_unico in ids_generados:
                    contador_ids_duplicados += 1
                    print(f"[{datetime.now()}] WARNING: ID duplicado detectado '{id_unico}'")
                    print(f"  Original: clave='{ids_generados[id_unico]['clave']}', país='{ids_generados[id_unico].get('pais', '')}'")
                    print(f"  Duplicado: clave='{row['clave']}', país='{row.get('pais', '')}'")
                    
                    # Generar un ID alternativo agregando un sufijo único
                    import time
                    id_unico = f"{id_unico}_{int(time.time() * 1000000) % 1000000}"
                    print(f"  Nuevo ID generado: '{id_unico}'")
                
                ids_generados[id_unico] = row
                ids_validos_bd.add(id_unico)
                
                # Crear texto para embedding
                clave_original = row['clave']
                nombre = row.get('nombre', '')
                certificacion = row.get('certificacion', '')
                pais = row.get('pais', '')
                
                texto_partes = [
                    str(clave_original),
                    str(nombre),
                    str(certificacion),
                    f"País: {pais}" if pais and pais != 'Sin_Pais' else ""
                ]
                texto = ' '.join([parte for parte in texto_partes if parte and parte != 'None']).strip()
                
                if not texto:
                    contador_errores += 1
                    print(f"[{datetime.now()}] ERROR: Texto vacío para clave {clave_original}")
                    continue
                
                batch_data[id_unico] = {
                    'row': row,
                    'texto': texto
                }
                
            except Exception as e:
                contador_errores += 1
                print(f"[{datetime.now()}] ERROR procesando registro: {e}")
                continue
        
        if not batch_data:
            print(f"[{datetime.now()}] No hay datos válidos en este lote")
            continue
            
        # Obtener vectores existentes en batch
        try:
            batch_ids = list(batch_data.keys())
            print(f"[{datetime.now()}] Consultando {len(batch_ids)} vectores existentes en Pinecone...")
            existentes_response = index.fetch(ids=batch_ids, namespace="Cursos")
            vectores_existentes = {}
            
            if existentes_response and hasattr(existentes_response, 'vectors') and existentes_response.vectors:
                for vec_id, vec_data in existentes_response.vectors.items():
                    if vec_data and hasattr(vec_data, 'metadata') and vec_data.metadata:
                        vectores_existentes[vec_id] = vec_data.metadata.get("hash")
                        
        except Exception as e:
            print(f"[{datetime.now()}] ERROR en fetch de Pinecone: {e}")
            vectores_existentes = {}
        
        # Procesar cada registro del lote
        # Procesar cada registro del lote
        for i, (id_unico, data) in enumerate(batch_data.items()):
            try:
                row = data['row']

                # Generar context y lc_id como cadena completa separada por ;
                valores = [str(row[col]) if row[col] is not None else "" for col in row.keys()]
                contexto_str = ";".join(valores)

                # Usar el contexto como texto a embeder
                texto = contexto_str

                hash_actual = generar_hash(texto)
                hash_remoto = vectores_existentes.get(id_unico)
                
                if hash_actual != hash_remoto:
                    # Generar embedding
                    vector = embed(texto)
                    
                    if not vector or len(vector) == 0:
                        contador_errores += 1
                        print(f"[{datetime.now()}] ERROR: Vector vacío para {id_unico}")
                        continue
                    
                    # Preparar metadata
                    metadata_limpia = {
                        "context": contexto_str,
                        "lc_id": contexto_str,
                        "orden": i + 1,
                        "tokens": contexto_str.split()
                    }
                    metadata_limpia["context"] = contexto_str
                    metadata_limpia["lc_id"] = contexto_str
                    metadata_limpia["orden"] = i + 1
                    metadata_limpia["tokens"] = contexto_str.split()
                    
                    vectores_para_upsert.append((id_unico, vector, metadata_limpia))
                    contador_actualizados += 1
                    
                    # Upsert en lotes de 50
                    if len(vectores_para_upsert) >= 50:
                        print(f"[{datetime.now()}] Haciendo upsert de {len(vectores_para_upsert)} vectores...")
                        try:
                            index.upsert(vectores_para_upsert, namespace="Cursos")
                            print(f"[{datetime.now()}] Upsert exitoso")
                        except Exception as upsert_error:
                            print(f"[{datetime.now()}] ERROR en upsert: {upsert_error}")
                            contador_errores += len(vectores_para_upsert)
                        vectores_para_upsert = []
                else:
                    contador_sin_cambios += 1
                    
            except Exception as e:
                contador_errores += 1
                print(f"[{datetime.now()}] ERROR procesando {id_unico}: {e}")
                continue
    
    # Upsert vectores restantes
    if vectores_para_upsert:
        print(f"[{datetime.now()}] Haciendo upsert final de {len(vectores_para_upsert)} vectores...")
        try:
            index.upsert(vectores_para_upsert, namespace="Cursos")
            print(f"[{datetime.now()}] Upsert final exitoso")
        except Exception as e:
            print(f"[{datetime.now()}] ERROR en upsert final: {e}")
            contador_errores += len(vectores_para_upsert)
    
    conn.close()
    
    # Resumen detallado
    print(f"\n[{datetime.now()}] === RESUMEN SINCRONIZACIÓN ===")
    print(f"Total registros BD (SELECT DISTINCT): {len(registros)}")
    print(f"Registros válidos procesados: {len(registros_validos)}")
    print(f"Registros inválidos (clave vacía): {registros_invalidos}")
    print(f"IDs únicos generados: {len(ids_validos_bd)}")
    print(f"IDs duplicados detectados y corregidos: {contador_ids_duplicados}")
    print(f"Vectores actualizados: {contador_actualizados}")
    print(f"Vectores sin cambios: {contador_sin_cambios}")
    print(f"Errores: {contador_errores}")
    
    # Verificación de integridad
    perdidos = len(registros_validos) - len(ids_validos_bd)
    if perdidos > 0:
        print(f"\n⚠️ ADVERTENCIA: Se perdieron {perdidos} registros en el proceso")
    else:
        print(f"\n✅ ÉXITO: Todos los registros válidos fueron procesados")
        print(f"Registros BD válidos: {len(registros_validos)}")
        print(f"IDs únicos en Pinecone: {len(ids_validos_bd)}")
    
    return ids_validos_bd

def eliminar_cursos_obsoletos_mejorada(ids_validos_bd: set):
    print(f"\n[{datetime.now()}] === INICIANDO ELIMINACIÓN DE OBSOLETOS ===")
    print(f"IDs válidos en BD: {len(ids_validos_bd)}")
    
    # Obtener todos los IDs de Pinecone
    print(f"[{datetime.now()}] Obteniendo todos los IDs de Pinecone...")
    ids_pinecone = obtener_todos_ids_pinecone("Cursos")
    
    print(f"IDs en Pinecone: {len(ids_pinecone)}")
    
    # Identificar obsoletos
    ids_a_borrar = list(ids_pinecone - ids_validos_bd)
    
    if ids_a_borrar:
        print(f"[{datetime.now()}] Encontrados {len(ids_a_borrar)} cursos obsoletos")
        
        # Mostrar algunos ejemplos
        if len(ids_a_borrar) <= 10:
            print("IDs a eliminar:")
            for id_ in ids_a_borrar:
                print(f"  - {id_}")
        else:
            print("Primeros 10 IDs a eliminar:")
            for id_ in ids_a_borrar[:10]:
                print(f"  - {id_}")
        
        # Eliminar en lotes
        batch_size = 1000
        for i in range(0, len(ids_a_borrar), batch_size):
            batch = ids_a_borrar[i:i+batch_size]
            print(f"[{datetime.now()}] Eliminando lote {i//batch_size + 1}: {len(batch)} vectores")
            
            try:
                index.delete(ids=batch, namespace="Cursos")
                print(f"[{datetime.now()}] Lote eliminado exitosamente")
            except Exception as e:
                print(f"[{datetime.now()}] ERROR eliminando lote: {e}")
        
        print(f"[{datetime.now()}] Eliminación completada.")
    else:
        print(f"[{datetime.now()}] No hay cursos obsoletos que eliminar.")

def obtener_todos_ids_pinecone(namespace):
    """Función mejorada para obtener todos los IDs de Pinecone usando el SDK"""
    ids = set()
    
    try:
        # Método 1: Usar stats para verificar si hay datos
        print(f"[{datetime.now()}] Verificando estadísticas del índice...")
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        namespace_count = stats.namespaces.get(namespace, {}).get('vector_count', 0)
        
        print(f"[{datetime.now()}] Total vectores en índice: {total_vectors}")
        print(f"[{datetime.now()}] Vectores en namespace '{namespace}': {namespace_count}")
        
        if namespace_count == 0:
            print(f"[{datetime.now()}] No hay vectores en el namespace '{namespace}'")
            return ids
            
        # Método 2: Usar query con vector dummy para obtener IDs
        print(f"[{datetime.now()}] Obteniendo IDs usando query...")
        
        # Crear un vector dummy para la query
        dummy_vector = [0.0] * 1536  # text-embedding-3-small tiene 1536 dimensiones
        
        # Query con top_k alto para obtener muchos IDs
        batch_size = 10000
        retrieved_count = 0
        
        while retrieved_count < namespace_count:
            try:
                query_response = index.query(
                    vector=dummy_vector,
                    top_k=min(batch_size, namespace_count - retrieved_count),
                    namespace=namespace,
                    include_metadata=False,
                    include_values=False
                )
                
                if not query_response.matches:
                    print(f"[{datetime.now()}] No se encontraron más matches")
                    break
                
                batch_ids = [match.id for match in query_response.matches]
                for id_ in batch_ids:
                    ids.add(id_)
                
                print(f"[{datetime.now()}] Batch: {len(batch_ids)} IDs, Total acumulado: {len(ids)}")
                retrieved_count += len(batch_ids)
                
                # Si obtuvimos menos de lo esperado, probablemente terminamos
                if len(batch_ids) < batch_size:
                    break
                    
            except Exception as e:
                print(f"[{datetime.now()}] ERROR en query batch: {e}")
                break
                
    except Exception as e:
        print(f"[{datetime.now()}] ERROR obteniendo estadísticas: {e}")
        
        # Método 3: Fallback usando list_paginated (si está disponible)
        try:
            print(f"[{datetime.now()}] Intentando método alternativo...")
            
            # Usar el método list del SDK si está disponible
            for ids_batch in index.list_paginated(namespace=namespace):
                batch_count = 0
                for id_ in ids_batch:
                    ids.add(id_)
                    batch_count += 1
                print(f"[{datetime.now()}] Batch alternativo: {batch_count} IDs")
                
        except AttributeError:
            print(f"[{datetime.now()}] Método list_paginated no disponible")
        except Exception as e:
            print(f"[{datetime.now()}] ERROR en método alternativo: {e}")
    
    print(f"[{datetime.now()}] Total final de IDs obtenidos: {len(ids)}")
    return ids

def verificar_sincronizacion_final():
    """Verificación final de la sincronización"""
    print(f"\n[{datetime.now()}] === VERIFICACIÓN FINAL DE SINCRONIZACIÓN ===")
    
    # Obtener stats actuales de Pinecone
    try:
        stats = index.describe_index_stats()
        namespace_count = stats.namespaces.get("Cursos", {}).get('vector_count', 0)
        print(f"Vectores actualmente en Pinecone namespace 'Cursos': {namespace_count}")
        
        # Obtener conteo actual de BD usando el mismo query
        conn = conectar_sql()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT COUNT(*) FROM (
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
        ) AS subconsulta
        """)
        
        conteo_bd_unico = cursor.fetchone()[0]
        conn.close()
        
        print(f"Registros únicos esperados en BD: {conteo_bd_unico}")
        diferencia = conteo_bd_unico - namespace_count
        
        if diferencia == 0:
            print("✅ SINCRONIZACIÓN PERFECTA: BD y Pinecone están 100% sincronizados")
        elif abs(diferencia) <= 5:  # Tolerancia pequeña
            print(f"✅ SINCRONIZACIÓN EXITOSA: Diferencia mínima de {diferencia} registros (dentro del margen aceptable)")
        else:
            print(f"⚠️ SINCRONIZACIÓN INCOMPLETA: Diferencia de {diferencia} registros")
            if diferencia > 0:
                print("   -> Hay registros de BD que no se sincronizaron a Pinecone")
            else:
                print("   -> Hay más registros en Pinecone que en BD")
        
        return namespace_count, conteo_bd_unico, diferencia
        
    except Exception as e:
        print(f"ERROR en verificación final: {e}")
        return 0, 0, 0

def debug_query_comparacion():
    """Función para debuggear y comparar los resultados del query original vs el nuevo"""
    print(f"[{datetime.now()}] === DEBUG COMPARACIÓN DE QUERIES ===")
    
    conn = conectar_sql()
    cursor = conn.cursor()
    
    # Query con SELECT DISTINCT (el que quieres usar)
    query_distinct = """
    SELECT DISTINCT  
        Ch.Clave AS clave,
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
    ORDER BY Ch.Clave, Vcc.Pais
    """
    
    try:
        cursor.execute(query_distinct)
        resultados = cursor.fetchall()
        
        print(f"Total registros con SELECT DISTINCT: {len(resultados)}")
        
        # Mostrar algunos ejemplos
        print("Primeros 10 registros (clave, país):")
        for i, (clave, pais) in enumerate(resultados[:10]):
            print(f"  {i+1}. Clave: '{clave}' | País: '{pais or 'NULL'}'")
        
        # Análisis por país
        paises_count = {}
        claves_sin_pais = 0
        
        for clave, pais in resultados:
            if not pais:
                claves_sin_pais += 1
                pais = 'Sin_Pais'
            
            if pais not in paises_count:
                paises_count[pais] = 0
            paises_count[pais] += 1
        
        print(f"\nDistribución por países:")
        for pais, count in sorted(paises_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pais}: {count} registros")
        
        print(f"\nRegistros sin país (NULL): {claves_sin_pais}")
        
    except Exception as e:
        print(f"ERROR en debug: {e}")
    finally:
        conn.close()

# Main simplificado y corregido
if __name__ == "__main__":
    print(f"[{datetime.now()}] === INICIO SINCRONIZACIÓN CURSOS CORREGIDO ===")
    
    # Debug opcional para verificar los datos
    print(f"\n[{datetime.now()}] === ANÁLISIS PRELIMINAR ===")
    debug_query_comparacion()
    
    print(f"\n[{datetime.now()}] === SINCRONIZACIÓN ===")
    ids_validos_cursos = sync_cursos()
    eliminar_cursos_obsoletos_mejorada(ids_validos_cursos)
    
    # Verificación final
    print(f"\n[{datetime.now()}] === VERIFICACIÓN POST-SINCRONIZACIÓN ===")
    pinecone_count, bd_count, diferencia = verificar_sincronizacion_final()
    
    print(f"\n[{datetime.now()}] === RESUMEN FINAL ===")
    print(f"Registros esperados (BD): {bd_count}")
    print(f"Registros en Pinecone: {pinecone_count}")
    print(f"IDs únicos generados durante sync: {len(ids_validos_cursos)}")
    print(f"Diferencia final BD vs Pinecone: {diferencia}")
    
    if diferencia == 0:
        print("🎉 SINCRONIZACIÓN COMPLETA Y EXITOSA")
    else:
        print("📋 REVISAR: La sincronización necesita ajustes adicionales")
    
    print(f"[{datetime.now()}] === SINCRONIZACIÓN COMPLETA ===")