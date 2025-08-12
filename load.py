
# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import os
from dotenv import find_dotenv, load_dotenv
from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()

dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
else:    
    raise FileNotFoundError("No .env file found in root directory of repository")
"""
if __name__ == "__main__":
    sql_query= "
    SELECT 
    ch.link_temario
FROM 
    cursos_habilitados ch 
JOIN 
    tecnologias t ON ch.tecnologia_id = t.id 
JOIN 
    complejidades c ON ch.complejidad_id = c.id 
JOIN 
    tipo_cursos tc ON ch.tipo_curso_id = tc.id 
JOIN 
    monedas m ON ch.moneda_id = m.id
JOIN
    cursos_estatus ce ON ch.curso_estatus_id = ce.id
JOIN
    familias f ON ch.familia_id = f.id
JOIN
    lineas_negocio ln ON ch.linea_negocio_id = ln.id
WHERE 
    ch.disponible = 1 
    AND (ce.nombre = 'Es Rentable' OR ce.nombre = 'Liberado')
    AND tc.nombre IN ('Intensivo', 'Programa', 'Digital')
    AND ch.link_temario IS NOT NULL 
    AND ch.link_temario <> ''
    AND ch.link_temario LIKE '%.pdf'

UNION ALL

SELECT 
    ch.link_temario
FROM 
    cursos_habilitados ch 
JOIN 
    tecnologias t ON ch.tecnologia_id = t.id 
JOIN 
    complejidades c ON ch.complejidad_id = c.id 
JOIN 
    tipo_cursos tc ON ch.tipo_curso_id = tc.id 
JOIN 
    monedas m ON ch.moneda_id = m.id
JOIN
    cursos_estatus ce ON ch.curso_estatus_id = ce.id
JOIN
    familias f ON ch.familia_id = f.id
JOIN
    lineas_negocio ln ON ch.linea_negocio_id = ln.id
WHERE 
    ch.subcontratado = 1 
    AND tc.nombre IN ('Intensivo', 'Programa', 'Digital')
    AND (ce.nombre = 'Es Rentable' OR ce.nombre = 'Liberado')
    AND ch.link_temario IS NOT NULL 
    AND ch.link_temario <> ''
    AND ch.link_temario LIKE '%.pdf'

UNION ALL

SELECT 
    ch.link_temario
FROM 
    cursos_habilitados ch 
JOIN 
    tecnologias t ON ch.tecnologia_id = t.id 
JOIN 
    complejidades c ON ch.complejidad_id = c.id 
JOIN 
    tipo_cursos tc ON ch.tipo_curso_id = tc.id 
JOIN 
    monedas m ON ch.moneda_id = m.id
JOIN
    cursos_estatus ce ON ch.curso_estatus_id = ce.id
JOIN
    familias f ON ch.familia_id = f.id
JOIN
    lineas_negocio ln ON ch.linea_negocio_id = ln.id
WHERE 
    ce.nombre IN ('Enviado a Operaciones', 'Enviado a Finanzas', 'Enviado a Comercial', 'Es Rentable')
    AND tc.nombre IN ('Intensivo', 'Programa', 'Digital')
    AND ch.link_temario IS NOT NULL 
    AND ch.link_temario <> ''
    AND ch.link_temario LIKE '%.pdf';"
"""
if __name__ == "__main__":
    sql_query= """
        SELECT 
            COALESCE(NULLIF(ch.clave, ''), 'NA') AS clave,
            COALESCE(NULLIF(ch.link_temario,''), 'Temario no encontrado') AS link_temario
        FROM 
            cursos_habilitados ch 
        JOIN 
            tecnologias t ON ch.tecnologia_id = t.id 
        JOIN 
            complejidades c ON ch.complejidad_id = c.id 
        JOIN 
            tipo_cursos tc ON ch.tipo_curso_id = tc.id 
        JOIN 
            monedas m ON ch.moneda_id = m.id
        JOIN
            cursos_estatus ce ON ch.curso_estatus_id = ce.id
        JOIN
            familias f ON ch.familia_id = f.id
        JOIN
            lineas_negocio ln ON ch.linea_negocio_id = ln.id
        WHERE 
            ch.disponible = 1 
            AND (ce.nombre = 'Es Rentable' OR ce.nombre = 'Liberado')
            AND tc.nombre IN ('Intensivo', 'Programa', 'Digital')
            AND ch.link_temario IS NOT NULL 
            AND ch.link_temario <> ''
            AND ch.link_temario LIKE '%.pdf'
        UNION ALL
        SELECT 
            COALESCE(NULLIF(ch.clave, ''), 'NA') AS clave,
            COALESCE(NULLIF(ch.link_temario,''), 'Temario no encontrado') AS link_temario
        FROM 
            cursos_habilitados ch 
        JOIN 
            tecnologias t ON ch.tecnologia_id = t.id 
        JOIN 
            complejidades c ON ch.complejidad_id = c.id 
        JOIN 
            tipo_cursos tc ON ch.tipo_curso_id = tc.id 
        JOIN 
            monedas m ON ch.moneda_id = m.id
        JOIN
            cursos_estatus ce ON ch.curso_estatus_id = ce.id
        JOIN
            familias f ON ch.familia_id = f.id
        JOIN
            lineas_negocio ln ON ch.linea_negocio_id = ln.id
        WHERE 
            ch.subcontratado = 1 
            AND tc.nombre IN ('Intensivo', 'Programa', 'Digital')
            AND (ce.nombre = 'Es Rentable' OR ce.nombre = 'Liberado')
            AND ch.link_temario IS NOT NULL 
            AND ch.link_temario <> ''
            AND ch.link_temario LIKE '%.pdf'
        UNION ALL
        SELECT 
            COALESCE(NULLIF(ch.clave, ''), 'NA') AS clave,
            COALESCE(NULLIF(ch.link_temario,''), 'Temario no encontrado') AS link_temario
        FROM 
            cursos_habilitados ch 
        JOIN 
            tecnologias t ON ch.tecnologia_id = t.id 
        JOIN 
            complejidades c ON ch.complejidad_id = c.id 
        JOIN 
            tipo_cursos tc ON ch.tipo_curso_id = tc.id 
        JOIN 
            monedas m ON ch.moneda_id = m.id
        JOIN
            cursos_estatus ce ON ch.curso_estatus_id = ce.id
        JOIN
            familias f ON ch.familia_id = f.id
        JOIN
            lineas_negocio ln ON ch.linea_negocio_id = ln.id
        WHERE 
            ce.nombre IN ('Enviado a Operaciones', 'Enviado a Finanzas', 'Enviado a Comercial', 'Es Rentable')
            AND tc.nombre IN ('Intensivo', 'Programa', 'Digital')
            AND ch.link_temario IS NOT NULL 
            AND ch.link_temario <> ''
            AND ch.link_temario LIKE '%.pdf';
        """

    hsr.pdf_loader (sql_query, namespace="Temarios")

