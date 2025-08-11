# -*- coding: utf-8 -*-
"""A class to manage the lifecycle of Pinecone vector database indexes."""

# document loading
import glob
import requests
import os
#from PyPDF2 import PdfReader


# general purpose imports
import json
import tempfile
import logging
from io import BytesIO
import pyodbc
# pinecone integration
import pinecone
from pinecone import Pinecone,PodSpec
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from langchain_community.document_loaders import PyPDFLoader #from langchain.document_loaders import PyPDFLoader
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document #from langchain.text_splitter import Document
from langchain_community.vectorstores import Pinecone as LCPinecone #from langchain.vectorstores.pinecone import Pinecone as LCPinecone

# this project
from models.conf import settings
#import fitz

logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.ERROR)

def download_pdf(url):
    # Descargar el PDF desde la URL
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error al descargar el PDF")

  

# pylint: disable=too-few-public-methods
class TextSplitter:
    """
    Custom text splitter that adds metadata to the Document object
    which is required by PineconeHybridSearchRetriever.
    """

    def create_documents(self, texts):
        """Create documents"""
        documents = []
        for text in texts:
            # Create a Document object with the text and metadata            
            document = Document(page_content=text, metadata={"context":text})            
            documents.append(document)
        return documents
    
    
class PineconeIndex:
    """Pinecone helper class."""

    _index: pinecone.Pinecone.Index = None
    _index_name: str = None
    _text_splitter: TextSplitter = None
    _openai_embeddings: OpenAIEmbeddings = None
    _vector_store: LCPinecone = None


    
    def __init__(self, index_name: str = "rag2"):
       
        self._pinecone_instance=Pinecone(
            api_key=settings.pinecone_api_key.get_secret_value(),
            host=#'https://rag-wib1770.svc.eastus-azure.pinecone.io'
            'https://rag2-wib1770.svc.eastus-azure.pinecone.io'
            
            )        
        
        self.index_name = index_name
        logging.debug("PineconeIndex initialized.")
           
        self.message_history=[]
    def add_to_history(self,message:BaseMessage):
        self.message_history.append(message)
    def get_history(self):
        return self.message_history

    @property
    def index(self) -> pinecone.Pinecone.Index:
        """pinecone.Index lazy read-only property."""
        if self._index is None:
            self._index=self.init_index()
            self._index =self._pinecone_instance.Index(name=self.index_name, host='https://rag2-wib1770.svc.eastus-azure.pinecone.io')       
        return self._index

    @property
    def index_stats(self) -> dict:
        """index stats."""
        retval = self.index.describe_index_stats()
        return json.dumps(retval.to_dict(), indent=4)

    @property
    def vector_store(self) -> LCPinecone:
        """Pinecone lazy read-only property."""
        if self._vector_store is None:
            self._vector_store = LCPinecone(
                index=self.index,
                embedding=self.openai_embeddings,
                text_key=settings.pinecone_vectorstore_text_key,
            )
        return self._vector_store

    @property
    def openai_embeddings(self) -> OpenAIEmbeddings:
        if self._openai_embeddings is None:
        
        #"""OpenAIEmbeddings lazy read-only property."""
            self._openai_embeddings=OpenAIEmbeddings(
                model="text-embedding-3-small",            
                dimensions=1536,
                api_key=settings.openai_api_key.get_secret_value(),
                organization=settings.openai_api_organization,     
            )
        return self._openai_embeddings

    @property
    def text_splitter(self) -> TextSplitter:
        """TextSplitter lazy read-only property."""
        if self._text_splitter is None:
            self._text_splitter = TextSplitter()
        return self._text_splitter

    def init_index(self):
        pc=Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        indexes= pc.list_indexes()
        print("Available indexes:", indexes) 

        if self.index_name not in indexes.names():
            print("Indexes does not exist. Creating...")
            pc.create_index(
                name=self.index_name,
                dimension=settings.pinecone_dimensions,
                metric="dotproduct",
                spec=PodSpec(
                    environment="eastus-azure"
                )
            )
            print("Index created")
            
        else:
            print("Index '{self.index_name}'already exists.")    

    def tokenize(self,text,prioritized_columns=None):
        tokens=[]
        if text is not None:
            return text.split()
        if prioritized_columns:
            for column in prioritized_columns:
                tokens.extend(column.split())
        return tokens

    def pdf_loader(self, conn):
        """
        Embed PDF from SQL da|base
        1. Connect to SQL database to retrieve PDF links
        2. Extract text from each PDF url
        3. Split into pages
        4.Embed each page 
        5. Store in Pinecone (upsert)
        """
        #self.initialize()
        connectionString =("DRIVER={ODBC Driver 18 for SQL Server};""SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;" "DATABASE=netec_prod;""UID=netec_read;""PWD=R3ad55**N3teC2025""TrustServerCertificate=yes;")        
        conn=pyodbc.connect(connectionString)
        cursor=conn.cursor()   
        #ejecutar consulta SQL
        sql_1="""
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
        cursor.execute(sql_1)
        pdf_links=cursor.fetchall()
        base_url="https://sce.netec.com/"
        i=0
        
        for pdf_link in pdf_links:
            i +=1
            j=len(pdf_links)
            pdf_url=pdf_link[0]
            print("PDF URL:", pdf_url)
            complete_url="https://sce.netec.com/"+pdf_url
            print(f"Downloading PDF {i} of {j}: {complete_url}")        
            loader=PyPDFLoader(file_path=complete_url)            
            docs=loader.load()
            k=0
            for doc in docs:
                k+=1
                print(k * "-", end="\r")
                documents = self.text_splitter.create_documents([doc.page_content])
                document_texts = [doc.page_content for doc in documents]
                embeddings = self.openai_embeddings.embed_documents(document_texts)
                self.vector_store.add_documents(documents=documents, embeddings=embeddings)
        print("Finished loading PDFs. \n" + self.index_stats)



    
    def load_sql(self,sql):
        """
        Load data from SQL database
        """
        self.initialize()
        #Establecer conexiÃ³n a la base de datos
        connectionString =("DRIVER={ODBC Driver 18 for SQL Server};""SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;" "DATABASE=netec_prod;""UID=netec_read;""PWD=R3ad55**N3teC2025""TrustServerCertificate=yes;")
        
        conn=pyodbc.connect(connectionString)
        cursor=conn.cursor()
              

        #ejecutar consulta SQL

        cursos="""
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
"""

        sql_statement_1="""
    SELECT DISTINCT
    Ch.Clave AS clave,
    Ch.Nombre AS nombre,
    Vcc.Clave AS clave_Examen,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IN ('Examen', 'Equipo') OR Vcc.Curso_Tipo_Elemento IS NULL
        THEN Vcc.Curso_Tipo_Elemento
        ELSE NULL
    END AS tipo_elemento,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IS NULL THEN NULL
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificaciÃ³n'
        WHEN Vcc.Tipo_Examen = 'CertificaciÃ³n' THEN Vcc.Tipo_Examen
        WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' THEN Vcc.Nombre_Catalogo
        ELSE NULL
    END AS nombre_examen,
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
    LEFT OUTER JOIN cursos_estatus ce
        ON Ch.curso_estatus_id = ce.id
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
    AND Vcc.Curso_Tipo_Elemento = 'Examen' -- CondiciÃ³n para tipo_elemento = 'Examen'
    AND Vcc.Tipo_Examen = 'certificaciÃ³n'; -- CondiciÃ³n para nombre_examen = 'certificaciÃ³n'

   
"""
        laboratorios="""
    SELECT DISTINCT
    Ch.Clave AS clave,
    Ch.Nombre AS nombre,
    Vcc.Clave AS clave_Examen,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IN ('Examen', 'Equipo') OR Vcc.Curso_Tipo_Elemento IS NULL
        THEN Vcc.Curso_Tipo_Elemento
        ELSE NULL
    END AS tipo_elemento,
    CASE
        WHEN Vcc.Curso_Tipo_Elemento IS NULL THEN NULL
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificaciÃ³n'
        WHEN Vcc.Tipo_Examen = 'CertificaciÃ³n' THEN Vcc.Tipo_Examen
        WHEN Vcc.Nombre_Catalogo LIKE 'Laboratorio%' THEN Vcc.Nombre_Catalogo
        ELSE NULL
    END AS nombre_examen,
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
    LEFT OUTER JOIN cursos_estatus ce
        ON Ch.curso_estatus_id = ce.id
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
    AND Vcc.Nombre_Catalogo LIKE 'Laboratorio%'
    """
        cursor.execute(cursos)
        rows=cursor.fetchall()
        columns=[column[0] for column in cursor.description]
        
        if 'link_temario' in columns:
            link_id=columns.index('link_temario')
        else:
            link_id=None
        
        prioritized_columns=['tecnologia_id', 'familia_id','nombre','clave']
       
        #Procesar cada fila y crear documentos
        for i,row in enumerate(rows):
            # Crear contenido con delimitadores
            content = ";".join(str(row[columns.index(col)]) if col in columns and row[columns.index(col)] is not None else '' for col in columns)
        
        # Crear contenido priorizado
            prioritized_content = ";".join(str(row[columns.index(col)]) if col in columns and row[columns.index(col)] is not None else '' for col in prioritized_columns if col in columns)
        
            tokens = self.tokenize(content, prioritized_columns=[prioritized_content])
            
            #Modificar los tokens para incluir el enlace del temario si existe
            if link_id is not None:
                link_temario=row[link_id]
                if link_temario and link_temario.strip() != "Temario no encontrado":
                    link="https://sce.netec.com/"+link_temario.strip()
                    tokens.append(link)
                    content += "|"+link
                else:
                    tokens.append(link_temario.strip())
                    content += "|"+link_temario.strip()
            document=Document(
                page_content=content,
                metadata={
                    "context": content,
                    "tokens":tokens,
                    "orden":i,
                    
                                    })

        #Embed the document
            embeddings=self.openai_embeddings.embed_documents([content])
            self.index.upsert(documents=[document],embeddings=embeddings)
            
            #self.vector_store.add_documents(documents=[document], embeddings=embeddings)
        print("Finished loading data from SQL. \n"+ self.index_stats)
        conn.close()