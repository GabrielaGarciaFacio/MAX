# # -*- coding: utf-8 -*-
# """
# Hybrid Search Retriever. A class that combines the following:
#     - OpenAI prompting and ChatModel
#     - PromptingWrapper
#     - Vector embedding with Pinecone
#     - Hybrid Retriever to combine vector embeddings with text search
 
# Provides a pdf loader program that extracts text, vectorizes, and
# loads into a Pinecone dot product vector database that is dimensioned
# to match OpenAI embeddings.
 
# See: https://python.langchain.com/docs/modules/model_io/llms/llm_caching
#      https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
#      https://python.langchain.com/docs/integrations/retrievers/pinecone_hybrid_search
# """
# # general purpose imports
import traceback
import logging
import datetime
from typing import Union
import pyodbc
import json
import urllib.parse
import uuid
from .logger import log_interaction

# pinecone integration
from langchain_community.cache import InMemoryCache
import streamlit as st
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.pdf import PyPDFLoader
# embedding
from langchain.globals import set_llm_cache
 
# prompting and chat
#from langchain.llms.openai import OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

 
# hybrid search capability
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from pinecone_text.sparse import BM25Encoder  # pylint: disable=import-error

from .conf import settings
#from models import pinecone
from models.pinecone_index import PineconeIndex
#from io import BytesIO
import tempfile
import requests
import re
from pydantic import BaseModel
import os

os.environ['PYTHONUTF8'] = '1'
class CustomPromptModel(BaseModel):
    prompt_template: PromptTemplate

    class Config:
        arbitrary_types_allowed = True

def download_pdf(url):
    # Descargar el PDF desde la URL
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error al descargar el PDF")

def load_full_text(url):
    # Descargar el PDF
    pdf_content = download_pdf(url)
    
    # Crear un archivo temporal para almacenar el PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_content)
        temp_pdf_path = temp_pdf.name

class Document:
  def __init__(self,page_content,metadata=None):
        self.page_content=page_content
        self.metadata=metadata if metadata is not None else {}

class HybridSearchRetriever:
    """Hybrid Search Retriever"""
 
    _chat: ChatOpenAI = None
    _b25_encoder: BM25Encoder = None
    _pinecone: PineconeIndex = None
    _retriever: PineconeHybridSearchRetriever = None
 
    def __init__(self):
        set_llm_cache(InMemoryCache())
        self.pinecone.message_history=[]
    def add_to_history(self,message:BaseMessage):
        self.pinecone.message_history.append(message)
    def format_history(self):
        # Limita el historial a las Ãºltimas n interacciones
        max_interactions = 5
        return " ".join([
            f"Usuario: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
            for msg in self.pinecone.message_history[-max_interactions:]
        ])
    def get_history(self):
        return self.pinecone.message_history  
         
       
    @property
    def pinecone(self) -> PineconeIndex:
        """PineconeIndex lazy read-only property."""
        if self._pinecone is None:
            self._pinecone = PineconeIndex()
        return self._pinecone
    '''
    # prompting wrapper
    @property
    def chat(self) -> ChatOpenAI:
        """ChatOpenAI lazy read-only property."""
        if self._chat is None:
            self._chat = ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),  # pylint: disable=no-member
                organization=settings.openai_api_organization,
                cache=settings.openai_chat_cache,
                max_retries=settings.openai_chat_max_retries,
                model=settings.openai_chat_model_name,
                temperature=settings.openai_chat_temperature,
            )
        return self._chat
'''
    @property
    def chat(self)->ChatOpenAI:
        if self._chat is None:
            self._chat=ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                organization=settings.openai_api_organization,
                cache=settings.openai_chat_cache,
                max_retries=settings.openai_chat_max_retries,
                model=settings.openai_chat_model_name,
                temperature=settings.openai_chat_temperature,
                model_kwargs={"stream": True}
            )
        return self._chat
  
    @property
    def bm25_encoder(self) -> BM25Encoder:
        #BM25Encoder lazy read-only property.
        if self._b25_encoder is None:
            self._b25_encoder = BM25Encoder().default()
        return self._b25_encoder
    """
    #@property
    def retriever(self, namespace="Cursos") -> PineconeHybridSearchRetriever:
        #PineconeHybridSearchRetriever lazy read-only property.
        if self._retriever is None:
            self._retriever = PineconeHybridSearchRetriever(
                embeddings=self.pinecone.openai_embeddings, 
                sparse_encoder=self.bm25_encoder, 
                index=self.pinecone.index,
                top_k=60,
                alpha=0.9,
                namespace=namespace)
        
        return self._retriever
    """
    def retriever(self, namespace="Cursos", top_k=40) -> PineconeHybridSearchRetriever:
        return PineconeHybridSearchRetriever(
            embeddings=self.pinecone.openai_embeddings,
            sparse_encoder=self.bm25_encoder,
            index=self.pinecone.index,
            top_k=top_k,
            alpha=0.9,
            namespace=namespace
        )

    def cached_chat_request(
        self, system_message: Union[str, SystemMessage], human_message: Union[str, HumanMessage]
    ) -> BaseMessage:
        """Cached chat request con soporte para transmisión."""
        if not isinstance(system_message, SystemMessage):
            logging.debug("Converting system message to SystemMessage")
            system_message = SystemMessage(content=str(system_message))

        if not isinstance(human_message, HumanMessage):
            logging.debug("Converting human message to HumanMessage")
            human_message = HumanMessage(content=str(human_message))
        
        messages = [SystemMessage(content=f"{self.format_history()}"),system_message, human_message]
        #print(f"\n\n---------------------------------------------MESSAGES----------------------------------------------------") #DEBUGGING
        #print(f"messagess: {messages}") #DEBUGGING
        #print("---------------------------------------------FIN MESSAGES-----------------------------------------------\n\n") #DEBUGGING
        response_content = ""
        if self.chat.stream:
            logging.debug("Chat en modo de transmisión")
            # Itera sobre los fragmentos de transmisión
            for chunk in self.chat.stream(messages):
                if isinstance(chunk, dict) and 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {}).get('content', '')
                    if delta:  # Solo agrega si el contenido no está vacío 
                        response_content += delta
                        #logging.debug(f"Received chunk: {delta}")
            #print(f"\n\n---------------------------------------------DELTA----------------------------------------------------") #DEBUGGING
            #print(f"Delta: {delta}") #DEBUGGING
            #print("---------------------------------------------FIN DELTA-----------------------------------------------\n\n") #DEBUGGING
            return HumanMessage(content=response_content)
        else:
            # Manejo de respuesta no transmitida
            retval = self.chat(messages)
            return retval

    def prompt_with_template(
        self, prompt: PromptTemplate, concept: str, model: str = settings.openai_prompt_model_name
    ) -> str:
        """Prompt with template."""
        llm = OpenAI(
            model=model,
            api_key=settings.openai_api_key.get_secret_value(),  # pylint: disable=no-member
            organization=settings.openai_api_organization
        )
        retval = llm(prompt.format(concept=concept))
        return retval
 
    def pdf_loader(self, conn, namespace="Temarios"):
        '''
        Embed PDF from SQL database
        1. Connect to SQL database to retrieve PDF links
        2. Extract text from each PDF url
        3. Split into pages
        4. Embed each page 
        5. Store in Pinecone (upsert)
        '''
    #self.initialize()
        connectionString = ("DRIVER={ODBC Driver 18 for SQL Server};"
                            "SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;"
                            "DATABASE=netec_prod;"
                            "UID=netec_read;"
                            "PWD=R3ad55**N3teC+*;"
                            "TrustServerCertificate=yes;")
        conn = pyodbc.connect(connectionString)
        cursor = conn.cursor()

        # Ejecutar consulta SQL optimizada
        sql_1 = """
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
            AND ch.link_temario LIKE '%.pdf';
            """
        
        
        cursor.execute(sql_1)
        pdf_links = cursor.fetchall()
        i = 0
        
        for pdf_link in pdf_links:
            i += 1
            j = len(pdf_links)
            pdf_url = pdf_link[0]
            #print("PDF URL:", pdf_url)
            complete_url = "https://sce.netec.com/" + pdf_url
            #print(f"Downloading PDF {i} of {j}: {complete_url}")
            
            loader = PyPDFLoader(file_path=complete_url)
            docs = loader.load()
            k = 0
            
            for doc in docs:
                k += 1
                print(k * "-", end="\r")
                documents = self.pinecone.text_splitter.create_documents([doc.page_content])
                document_texts = [doc.page_content for doc in documents]
                embeddings = self.pinecone.openai_embeddings.embed_documents(document_texts)
                self.pinecone.vector_store.add_documents(documents=documents, embeddings=embeddings, namespace="Temarios")

        #print("Finished loading PDFs. \n" + self.pinecone.index_stats)


    def safe_int_conversion(value,default=0):
            if isinstance(value,(str,int,float)):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default
 
    def extract_data_from_lc_id(self, lc_id_value,namespace="Cursos"):
        print("Ejecutando extract_data_from_lc_id...")  # Debugging básico
        print(f"lc_id_value recibido: {repr(lc_id_value)}")
        global values
        values = lc_id_value.split(';')
        print(f"Valores después de split: {values}")
        print(f"Cantidad de valores en lc_id_value: {len(values)}")

        #Si estamos en el namespace "Cursos" asignar los valores correctamente
        if namespace == "Cursos":
            keys = ["clave", "nombre", "certificacion", "disponible", "sesiones", "precio",
                    "subcontratado", "pre_requisitos", "tecnologia_id", "complejidad_id",
                    "tipo_curso_id", "nombre_moneda", "estatus_curso", "familia_id", "horas",
                    "link_temario", "linea_negocio_id", "version", "entrega", "clave_Examen",
                    "nombre_examen", "tipo_elemento", "base_costo", "pais", "costo"]
        else:  # Namespace "Labs" o "examenes"
            keys = ["clave", "nombre", "clave_examen", "tipo_elemento", "nombre_examen", "base_costo", "pais", "costo"]

         
        values.extend(['NA'] * (len(keys) - len(values)))
        extracted_data=dict(zip(keys,values))
        link_temario_value = extracted_data.get("link_temario", "").strip()        
        encoded_url = 'Temario no encontrado'
                
        if link_temario_value:
        #print(f"Link temario original: {link_temario_value}")  # Debugging line
            if link_temario_value.startswith("files/") or link_temario_value.startswith("https://"):
                complete_url="https://sce.netec.com/" + link_temario_value
                encoded_url=urllib.parse.quote(complete_url, safe='/:')
            elif link_temario_value.startswith("https://"):
                encoded_url=urllib.parse.quote(link_temario_value, safe='/:')
        extracted_data["link_temario"] = encoded_url
        print(f"Link generado: {encoded_url}")
        data = {
            "clave": values[0].strip() if len(values) > 0 else 'NA',
            "nombre": values[1].strip() if len(values) > 1 else 'NA',
            "certificacion": values[2].strip() if len(values) > 2 else 'NA',
            "disponible": values[3].strip() if len(values) > 3 else 'NA',
            "sesiones": values[4].strip() if len(values) > 4 else 'NA',
            "precio": values[5].strip() if len(values) > 5 else 'NA',
            "subcontratado": values[6].strip() if len(values) > 6 else 'NA',
            "pre_requisitos": values[7].strip() if len(values) > 7 else 'NA',
            "tecnologia_id": values[8].strip() if len(values) > 8 else 'NA',
            "complejidad_id": values[9].strip() if len(values) > 9 else 'NA',
            "tipo_curso_id": values[10].strip() if len(values) > 10 else 'NA',
            "nombre_moneda": values[11].strip() if len(values) > 11 else 'NA',
            "estatus_curso": values[12].strip() if len(values) > 12 else 'NA',
            "familia_id": values[13].strip() if len(values) > 13 else 'NA',
            "horas": values[14].strip() if len(values) > 14 else 'NA',
            "link_temario":encoded_url if len(values) > 15 else 'Temario no encontrado',
            "linea_negocio_id":values[16].strip() if len(values) > 16 else 'NA',
            "version": values[17].strip() if len(values) > 17 else 'NA',        
            "entrega": values[18].strip() if len(values) > 18 else 'NA',
            "clave_Examen":values[19].strip() if len (values)>19 else 'NA',
            "nombre_examen":values[20].strip() if len (values)>20 else 'NA',
            "tipo_elemento":values[21].strip() if len (values)>21 else 'NA',
            "tipo_examen":values[22].strip() if len (values)>22 else 'NA',
            "base_costo":values[23].strip() if len (values)>23 else 'NA',
            "pais":values[24].strip() if len (values)>24 else 'NA',
            "costo":values[25].strip() if len (values)>25 else 'NA',
        }
        

        print(json.dumps(extracted_data, indent=2, ensure_ascii=False))
        #return data
        return extracted_data

    #Load sql database
    def load_sql(self,sql,namespace="Labs"):
        #self.initialize()
       
        #Connect to the bd
        connectionString =("DRIVER={ODBC Driver 18 for SQL Server};""SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;" "DATABASE=netec_prod;""UID=netec_read;""PWD=R3ad55**N3teC+*;""TrustServerCertificate=yes;")
        conn=pyodbc.connect(connectionString)
        cursor=conn.cursor()
 
        #Execute the provided SQL command
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
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificación'
        WHEN Vcc.Tipo_Examen = 'Certificación' THEN Vcc.Tipo_Examen
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
    AND Vcc.Clave LIKE 'Lab-%'
    
"""    
        examenes="""

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
        WHEN Vcc.Curso_Tipo_Elemento = 'Examen' THEN 'certificación'
        WHEN Vcc.Tipo_Examen = 'Certificación' THEN Vcc.Tipo_Examen
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
    AND Vcc.Curso_Tipo_Elemento = 'Examen' -- Condición para tipo_elemento = 'Examen'
    AND Vcc.Tipo_Examen = 'certificación'; -- Condición para nombre_examen = 'certificación'
    """
        cursor.execute(laboratorios) 
        rows=cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        print(f"Columnas en la consulta SQL: {columns}")

        if 'link_temario' in columns:
            link_id = columns.index('link_temario')
        else:
            link_id = None

        prioritized_columns=['tecnologia_id', 'familia_id','nombre','clave']
              
        for i, row in enumerate(rows):
            row = tuple("NAA" if col is None else col for col in row)
            print(f"Row antes de construir lc_id: {row}")
            # Construir contenido
            content = ";".join(str(col) for col in row if col is not None)
            
            # Obtener datos de lc_id y asignar a la plantilla
            #lc_id_value = row[columns.index('lc_id')] if 'lc_id' in columns else ''
            lc_id_value = [str(col) for col in row]  # Convertimos cada elemento de la fila a string
            lc_id_value = ";".join(lc_id_value)  # Unimos los elementos con punto y coma
            #print(f"LC_ID value from row: {lc_id_value}") 
            template =self.extract_data_from_lc_id(lc_id_value)
                       
            # Convertir la plantilla a JSON y tokenizar
            template_str = json.dumps(template, ensure_ascii=False)
            #print(f"Template JSON: {template_str}")  # Debugging line
            tokens = self.pinecone.tokenize(template_str, prioritized_columns=prioritized_columns)

            document=Document(
                page_content=content, #contents
                metadata={
                    "context":content,
                    "tokens":tokens,
                    "orden":i,
                    "lc_id": lc_id_value
                    
            })
            setattr(document, "id", str(uuid.uuid4()))
            embeddings=self.pinecone.openai_embeddings.embed_documents([content])
            self.pinecone.vector_store.add_documents(documents=[document],embeddings=embeddings, namespace="Laboratorios")
       
        #print("Finished loading data from SQL "+ self.pinecone.index_stats)
        conn.close()

    def clasificar_intencion_con_gpt(self, consulta: str) -> str:
        prompt_clasificacion = f"""Clasifica el siguiente mensaje según su intención:
        
        Consulta: "{consulta}"
        
        Responde solo con "Cursos" si la consulta pregunta sobre un curso en general (por ejemplo, precio, disponibilidad, certificaciÃ³n, sesiones, etc.) ; en caso de que la consulta diga que curso esta relacionado a cierta certificacion o sólo se mencione el nombre de un posible curso (por ejemplo: java).
        Responde solo con "Certificaciones" si la consulta solicita certificaciones de algún tema o fabricante, no aplica si la consulta dice que curso esta asociado a cierta certificacion. 
        Responde solo con "Precio" si la consulta solicita precio o precios de cursos.
        Responde solo con "General" si la consulta es sobre un tema en general sin mencionar una clave especifica.
        Responde solo con "Agnostico" si la consulta hace referencia a la palabra agnostico o alguna de sus variantes o conjugaciones.
        Responde solo con "Temarios" si el mensaje pregunta especí­ficamente por el temario, contenido, de que trata uno o mas cursos.
        Responde solo con "Recomendacion" si el mensaje solicita algún curso para cubrir algún tema o grupo de temas, por ejempo ("Cursos de administración de proyectos y agile")
        Responde solo con "No" si la consulta no tiene relación con estos temas, si es demasiado general para determinar una intención específica o si no se cuenta con la información.
        Responde solo con "Laborarorio" si la consulta solicita información correspondiente a los laboratorios de los cursos (por ejemplo, precio, disponibilidad, país, clave, nombre)
        Responde solo con "Examenes" si la consulta solicita información correspondiente a los exámenes de los cursos (por ejemplo, precio, disponibilidad, país, clave, nombre)
        Intención:"""
        #        Responde solo con "Variada" si la consulta contiene varios tema de TI.
        chat = ChatOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        organization=settings.openai_api_organization,
        cache=settings.openai_chat_cache,
        max_retries=settings.openai_chat_max_retries,
        model=settings.openai_chat_model_name,
        temperature=settings.openai_chat_temperature
        )
        # Ejecuta la consulta con GPT-4 mini
        respuesta = chat([HumanMessage(content=prompt_clasificacion)])

        # Extrae la intenciÃ³n
        intencion = respuesta.content.strip()
        return intencion

    def rag(self, human_message: Union[str, HumanMessage],conversation_history=None):
        """
        Retrieval Augmented Generation prompt.
        1. Retrieve human message prompt: Given a user input, relevant splits are retrieved
           from storage using a Retriever.
        2. Generate: A ChatModel / LLM produces an answer using a prompt that includes
           the question and the retrieved data
 
        To prompt OpenAI's GPT-3 model to consider the embeddings from the Pinecone
        vector database, you would typically need to convert the embeddings back
        into a format that GPT-3 can understand, such as text. However, GPT-3 does
        not natively support direct input of embeddings.
 
        The typical workflow is to use the embeddings to retrieve relevant documents,
        and then use the text of these documents as part of the prompt for GPT-3.
        """
        def format_response(data, labs_data=None, examenes_data=None): 
            """
            Función mejorada que SIEMPRE respeta la plantilla unificada
            """
            # Validación inicial
            if not isinstance(data, dict):
                return "Error: Datos de curso no válidos"
            
            # Convertir tuplas a diccionarios si es necesario
            if isinstance(labs_data, tuple) and len(labs_data) >= 3:
                labs_data = {
                    "nombre_examen": labs_data[0], 
                    "clave_examen": labs_data[1], 
                    "costo": labs_data[2]
                }
            
            if isinstance(examenes_data, tuple) and len(examenes_data) >= 3:
                examenes_data = {
                    "nombre_examen": examenes_data[0], 
                    "clave_examen": examenes_data[1], 
                    "costo": examenes_data[2]
                }
            
            # Caso especial: cursos subcontratados
            if "(sub)" in data.get('clave', '').lower():
                return f"""**Curso Subcontratado**
        - **Clave**: {data.get('clave', 'NA')}
        - **Nombre**: {data.get('nombre', 'NA')}
        - **Más información**: [Subcontrataciones](https://netec.sharepoint.com/Subcontrataciones/subcontratacioneslatam/SitePages/Inicio.aspx)"""
            
            # Procesar link del temario
            link_temario = data.get('link_temario', 'Temario no encontrado')
            if link_temario and link_temario != 'Temario no encontrado':
                if link_temario.startswith("https://"):
                    link_temario = f"[Link al temario]({link_temario.replace(' ', '%20')})"
                else:
                    link_temario = 'Temario no encontrado'
            else:
                link_temario = 'Temario no encontrado'
            
            # Procesar información de laboratorios
            if labs_data and isinstance(labs_data, dict):
                labs_info = f"Sí\n  - Nombre: {labs_data.get('nombre_examen', 'NA')} [{labs_data.get('clave_examen', 'NA')}]\n  - Costo: {labs_data.get('costo', 'NA')}"
            else:
                labs_info = "No lleva laboratorio"
            
            # Procesar información de exámenes
            if examenes_data and isinstance(examenes_data, dict):
                examenes_info = f"Sí\n  - Nombre: {examenes_data.get('nombre_examen', 'NA')} [{examenes_data.get('clave_examen', 'NA')}]\n  - Costo: {examenes_data.get('costo', 'NA')}"
            else:
                examenes_info = "No tiene certificación asociada"
            
            # Determinar disponibilidad
            if data.get('disponible', 'NA') == 'Si' and data.get('estatus_curso', '') == 'Liberado':
                disponibilidad = "Habilitado"
            elif data.get('disponible', 'NA') == 'No' or data.get('estatus_curso', '') != 'Liberado':
                disponibilidad = "En habilitación"
            else:
                disponibilidad = "NA"
            
            # PLANTILLA UNIFICADA - ESTA ES LA ÚNICA FUENTE DE VERDAD
            response_template = f"""**Curso**
        - **Clave**: {data.get('clave', 'NA')}
        - **Nombre**: {data.get('nombre', 'NA')} V{data.get('version', 'NA')}
        - **Tecnología/ Línea de negocio**: {data.get('tecnologia_id', 'NA')} / {data.get('linea_negocio_id', 'NA')}
        - **Entrega**: {data.get('entrega', 'NA')}
        - **Estatus del curso**: {disponibilidad}
        - **Tipo de curso**: {data.get('tipo_curso_id', 'NA')}
        - **Sesiones y Horas**: {data.get('sesiones', 'NA')} / {data.get('horas', 'NA')}
        - **Precio**: {data.get('precio', 'NA')} {data.get('nombre_moneda', 'NA')}
        - **Examen**: {examenes_info}
        - **Laboratorio**: {labs_info}
        - **País**: {data.get('pais', 'NA')}
        - **Complejidad**: {data.get('complejidad_id', 'NA')}
        - **Link al temario**: {link_temario}"""
            
            return response_template
        


        if not isinstance(human_message, HumanMessage):
            logging.debug("Converting human_message to HumanMessage")
            human_message = HumanMessage(content=str(human_message))
 
        self.add_to_history(human_message)
        
        if conversation_history:
            context=" ".join([msg.content for msg in conversation_history[-5:]])
        else:
            context=""
        def ordenar_cursos(cursos):
            """
            Ordena los cursos en función de:
            - Complejidad (de menor a mayor)
            - Disponibilidad (primero los disponibles)
            - Tipo de curso (Intensivo > Digital > Programa)
            """
            complejidad_map = {
                'CORE': 1,
                'FOUNDATIONAL': 2,
                'BASIC': 3,
                'INTERMEDIATE': 4,
                'SPECIALIZED': 5,
                'ADVANCED': 6
            }

            tipo_curso_map = {
                'Intensivo': 1,
                'Digital': 2,
                'Programa': 3
            }
        
            def clave_ordenamiento(curso):
                data= curso.metadata
                return (
                    0 if data.get('disponibe','NA')=='Sí' else 1,
                    complejidad_map.get(curso.metadata.get('complejidad_id', 'ADVANCED'), 6),
                    tipo_curso_map.get(curso.metadata.get('tipo_curso_id', 'Programa'), 3)
                )

            return sorted(cursos,key=clave_ordenamiento)
        
        def mostrar_cursos_ordenados(cursos):
            """
            Muestra los cursos ordenados y agrupa los subcontratados en un bloque especial.
            """
            cursos_disponibles = [curso for curso in cursos if curso.metadata.get('disponible', 'No') == 'Si']
            cursos_subcontratados = [curso for curso in cursos if curso.metadata.get('subcontratado', 'No') == 'Si']
            cursos_habilitacion = [curso for curso in cursos if curso.metadata.get('estatus_curso', '') == 'En habilitación']
            if cursos_disponibles:
                print("\n**Cursos Disponibles:**")
                for curso in ordenar_cursos(cursos_disponibles):
                    data = self.extract_data_from_lc_id(curso.metadata.get('lc_id', ''), namespace="Cursos")
                    formatted_response = format_response(data)
                    print(formatted_response)

            if cursos_subcontratados:
                print("\n**En otras modalidades de entrega te ofrecemos:**")
                for curso in ordenar_cursos(cursos_subcontratados):
                    data = self.extract_data_from_lc_id(curso.metadata.get('lc_id', ''), namespace="Cursos")
                    formatted_response = format_response(data)
                    print(formatted_response)

            if cursos_habilitacion:
                print("\n**Cursos en habilitación:**")
                for curso in ordenar_cursos(cursos_habilitacion):
                    data = self.extract_data_from_lc_id(curso.metadata.get('lc_id', ''), namespace="Cursos")
                    formatted_response = format_response(data)
                    print(formatted_response)


        # ---------------------------------------------------------------------
        # 1.) Retrieve relevant documents from Pinecone vector database
        # ---------------------------------------------------------------------     
        def translate_with_mymemory(text):
            url = "https://api.mymemory.translated.net/get"
            params = {
                'q': text,
                'langpair': 'es|en'  # TraducciÃ³n de espaÃ±ol a inglÃ©s
            }

            response = requests.get(url, params=params)
            result = response.json()
            translated_text = result['responseData']['translatedText']
            return translated_text
           
        def clean_query(query):
            # Eliminar sÃ­mbolos de puntuaciÃ³n usando una expresiÃ³n regular
            query = re.sub(r'[Â¿?Â¡!]', '', query)  # Elimina los sÃ­mbolos especificados
            # Lista de palabras que no aportan valor a la bÃºsqueda
            stop_words = ["agnóstico","agnostico", "tenemos", "tienes", "tiene", "de", "a", "los", "curso", "cursos", "hay", "algún", "alguna", "precio", "presupuesto", "y", "para", "que", "en", "el", "catalogo", "catalogo?", "cubran", "cubra", "puedo", "ofrecer", "cliente", "quiere", "conocer", "principales", "principal", "elemento", "elementos", "un", "digital"] #, "a", "un", "los", "principales"
            # Divide la consulta en palabras y elimina las stop words
            cleaned_queryes = " ".join([word for word in query.lower().split() if word not in stop_words])
            # TraducciÃ³n de la consulta limpia al inglÃ©s
            cleaned_queryen = translate_with_mymemory(cleaned_queryes)
            # Devuelve la consulta limpia en espaÃ±ol y su traducciÃ³n en inglÃ©s
            if cleaned_queryes.strip() == cleaned_queryen.strip():
                return cleaned_queryes
            else:
                return f"{cleaned_queryes} {cleaned_queryen}"
        
        
        leader_certificaciones =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            IMPORTANTE: USA EXACTAMENTE la información formateada que se te proporciona. NO modifiques el formato.

            **Instrucciones para CERTIFICACIONES**:
            - Si la consulta incluye términos como "certificación" o "certificaciones", muestra SÓLO la lista de certificaciones disponibles
            - NO incluyas cursos donde el campo "Certificación" sea "Ninguna" o esté vacío
            - Cada certificación se menciona una sola vez (no repetir si varios cursos la comparten)
            - Si preguntan sobre disponibilidad de una certificación, responde y sugiere cursos asociados
            - NO incluyas cursos adicionales

            **Listado de cursos:**"""
        
        leader_precio =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            **Instrucciones para PRECIO**:
            - Si la consulta incluyes "precio" o "costo" y menciona un curso específico: proporciona ÚNICAMENTE el precio
            - Formato: "Nombre del curso: Precio Moneda" (Una línea por curso)
            - Si menciona examen o laboratorio específico: "Nombre del examen/laboratorio: Precio Moneda"
            - NO incluyas otra información (disponibilidad, sesiones, tecnología, etc.)
            - Incluye al menos 5 cursos con su precio cuando sea aplicable
            
            **Listado de cursos**"""
        
        leader_general =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            IMPORTANTE: USA EXACTAMENTE la información formateada que se te proporciona. NO modifiques el formato.

            **Instrucciones para CONSULTAS GENERALES**:
            - Si el usuario saluda, salúdalo y ofrece ayuda amablemente
            - Para temas/tecnologías específicas: comienza con "Estos son algunos cursos relacionados con tu consulta:"
            - Para recomendaciones a clientes: responde con todas las entregas del mismo curso (ej: CCNA, CCNA Digital, CCNA Digital-CLC)
            - Si hay cursos subcontratados: muestra información completa y añade "Te recomiendo ponerte en contacto con un Ing. preventa para más información"
            - Para servicios específicos (AWS Lambda, Kubernetes, etc.): añade al final: "Te sugiero ahondar más con un Ing. preventa"
            
            **FORMATO OBLIGATORIO PARA CADA CURSO**:
            1. Haz un resúmen humanizado usando información de **Temarios** (mínimo 150 palabras por curso)
            2. Inmediatamente después del resumen, DEBES mostrar la plantilla completa del curso que aparece en el listado
            3. Estructura: **Nombre del curso [Clave]** (en negritas) -> resumen ->plantilla completa del curso
            4. NO omitas ningún campo de la plantilla del curso
            5. SIEMPRE muestra tanto el resumen como la plantilla completa para cada curso

            Ejemplo de estructura:
            **Nombre del Curso [CLAVE]**
            [Resumen de 150+ palabras usando información de temrios]

            [Plantilla completa del curso tal como aparece en el listado - sin modificar]
            
            **Listado de cursos y temarios**"""
        
        leader_agnostico=f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que están en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            IMPORTANTE: USA EXACTAMENTE la información formateada que se te proporciona. NO modifiques el formato.

            **Instrucciones para CURSOS AGNÓSTICOS**:
            - Si incluye "agnóstico/agnósticos": muestra ÚNICAMENTE cursos que contengan "Comptia", "CCNA", "APMG" o "GKI"
            - NO muestres cursos con "Microsoft", "AWS", "Cisco", "ECCouncil", o "Palo Alto"
            - Haz resúmenes usando información de **Temarios** (mínimo 150 palabras por curso)
            - Formato: nombre del curso y clave en negritas, seguido del resumen, luego la plantilla del curso
            - Si no encuentras cursos agnósticos: "Disculpa, no tengo esta información. Favor de ponerte en contacto con un Ing. Preventa."

            **Listado de cursos y temarios**"""
        
        leader_temarios =f"""Eres un asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            
            **Instrucciones para TEMARIOS**:
            - Responde de manera humanizada la consulta del usuario
            - Si piden temario de un curso: proporciona el enlace con breve descripción
            - Ejemplo: "Puedes encontrar el temario del curso AZ-900T00 en: [enlace]"
            - USA la información de **Listado de temarios**

            **Listado de temarios:**"""
                      
                                          
        leader_recomendacion2 =f"""Eres un asistente útil,tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            IMPORTANTE: USA EXACTAMENTE la información formateada que se te proporciona. NO modifiques el formato.

            **Instrucciones para RECOMENDACIONES**:
            - Responde de manea humanizada usando información de **Listado de cursos y temarios**
            - ORDEN: Primero cursos habilitados, luego cursos en habilitación y al final cursos subcontratados
            - CRITERIO DE ORDENACIÓN
                * Por complejidad: CORE(1), FOUNDATIONAL(2), BASIC(3), INTERMEDIATE(4), SPECIALIZED(5), ADVANCED(6)
                * Por tipo de curso: Intensivo(1), Digital(2), Programa(3)
            - Para consultas sobre "versión" específica: responde sólo con esa información
            - Si existen varios cursos con clave similar: incluye TODOS (ej: CCNA, CCNA Digital, CCNA Digital-CLC)
            - Para temas desconocidos ("cursos confirmados", "tutorías", "desarrollos"): "Disculpa, no tengo esta información. Favor de contactar a un Ing. Preventa"
            - Haz resúmenes usando **Temarios** (mínimo 150 palabras por curso)
            - Formato: nombre del curso y clave en negritas, seguido del resumen, luego la plantilla del curso
            
            **Listado de cursos:**"""
        
        leader_labs =f"""Eres un asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.

            **Instrucciones para LABORATORIOS**:
            - Cuando pregunten por laboratorio de un curso: responde con nombre y clave entre corchetes
            - Ejemplo: "Sí, el curso tiene el laboratorio: Laboratorio rentado directamente con Micosoft [Lab-MIC-5]"
            - Para clave específica de laboratorio usa información de 'labs_data'
            - Ejemplo: "La clave del laboratorio es Lab-Mile2"
            - RESPONDE SÓLO lo que te pregunten

            **Listado de cursos:**"""
        
        leader_exa=f"""Eres un asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            
            **Instrucciones para EXÁMENES**:
            - Cuando pregunten por examen de un curso: responde con nombre y clave entre corchetes
            - Ejemplo: "Sí, para este curso encuentro: certificación [220-1102 or 220-2202]"
            - Usa información de 'examenes_data' para el nombre
            - IMPORTANTE: Examen de certificación no es igual que examen de curso
            - Si preguntan por examen de curso, responde con examen de curso, NO de cetificación
            - RESPONDE SÓLO LO QUE TE PREGUNTEN

            **Listado de cursos:**"""
        
        leader_No =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la informaciónn en español.

            **Instrucciones para CONSULTAS NO CLARAS**:
            - Si el usuario saluda: salúdalo y ofrece ayuda amablemente
            - Para consultas no claras: "¡Hola! Parece que tu consulta no está clara o no está relacionada con nuestros temas. ¿Podrías darme más detalles?"
            - Para información no disponible ("cursos confirmados", "tutorías", "desarrollos"): "Disculpa, no tengo esa información. Favor de ponerte en contacto con un Ing. Preventa"

            **Listado de cursos:**"""
        intencion = self.clasificar_intencion_con_gpt(human_message.content)
        print(f"intencion: {intencion}")
        
        # Selecciona el namespace y el prompt basados en la intención
        if intencion == "Cursos":
            namespace = "Cursos"
            leader = leader_recomendacion2
        elif intencion == "Certificaciones":
            namespace = "Cursos"
            leader = leader_certificaciones
        elif intencion == "Precio":
            namespace = "Cursos"
            leader = leader_precio
        elif intencion == "General":
            namespace = "Cursos"
            leader = leader_general
        elif intencion == "Agnostico":
            namespace = "Cursos"
            leader = leader_agnostico
        elif intencion == "Temarios":
            namespace = "Temarios"
            leader=leader_temarios
        elif intencion=="Laboratorio":
            namespace="Labs"
            leader = leader_labs
        elif intencion=="Examenes":
            namespace="Examenes"
            leader = leader_exa
        elif intencion == "Recomendacion":
            namespace = "Cursos" 
            leader = leader_recomendacion2
        else:   
            namespace="Cursos"
            leader=leader_No         

        enhanced_query2 = f"{context} {human_message.content}"
        enhanced_query = f"{clean_query(human_message.content)}"  
        if not enhanced_query.strip():
            print("Consulta vací­a, se omite la búsqueda en Pinecone.")
            return "Disculpa, no puedo procesar una consulta vaci­a. Por favor, intenta nuevamente."

        # Realizar la búsqueda solo si enhanced_query no estÃ¡ vacÃ­o
        try:
            documents = self.retriever(namespace=namespace).get_relevant_documents(query=enhanced_query)
            print(f"Documentos recuperados en {namespace}: {len(documents)}")
            
            for doc in documents:
                print(f"Documento en {namespace}: {doc.metadata.get('lc_id')}")
            documents = sorted(documents, key=lambda doc: doc.metadata.get("orden", 0))

        
        except Exception as e:
            print(f"Error al realizar la búsqueda en Pinecone: {e}")
            logging.error(f"Error al realizar la búsqueda en Pinecone: {e}")
            return "Disculpa, hubo un problema al realizar la búsqueda. Intenta nuevamente."
        
        #Realiza búsquedas adicionales en 'labs' y 'examenes si la intención es 'cursos'
        labs_documents = []
        exams_documents = []
        if intencion in ["Cursos","Labs","Examenes"]:
            #additional_namespace_2 = "Labs" if namespace == "Cursos" else "Examenes"
            try:
                labs_documents=self.retriever(namespace='Labs').get_relevant_documents(query=enhanced_query)
                exams_documents=self.retriever(namespace='Examenes').get_relevant_documents(query=enhanced_query)
                
            except Exception as e:
                print(f'Error al realizar la búsqueda en Pinecone: {e}')
                return 'Disculpa, hubo un problema al realizar la búsqueda. Intenta nuevamente'
        # Si no se encontraron exámenes, se hace una búsqueda alternativa con el nombre del curso 
            if not exams_documents:
                print("No se encontraron exámenes en la primera búsqueda. Intentando búsqueda alternativa")
                try:
                    exams_documents=self.retriever(namespace='Examenes').get_relevant_documents(query=data.get('nombre', ''))
            
                    print("\nResultados de búsqueda alternativa en Exámenes:")
                    for doc in exams_documents:
                        print(doc.metadata.get('lc_id'))
                    #Agregar los documenos encontrados en la búsqueda alternativa
                    documents.extend(exams_documents)
                except Exception as e:
                    print(f'Error al realizar la búsqueda en Pinecone: {e}')
           
            
            for doc in labs_documents:
                print(doc.metadata.get('lc_id'))             
    

            print("\nBuscando coincidencias en Examenes:")
            for doc in exams_documents:
                print(doc.metadata.get('lc_id'))
                

                

            documents.extend(labs_documents)
            documents.extend(exams_documents)
        

        if intencion == "Recomendacion" or intencion == "Cursos" or intencion == "Agnostico": 
            additional_namespace = "Temarios" if namespace == "Cursos" else "Cursos"
            additional_documents = self.retriever(namespace=additional_namespace).get_relevant_documents(query=enhanced_query)
            # Extrae solo las claves de `lc_id` en los documentos adicionales
            def extraer_clave(document):
                match = re.search(r"Clave:\s*\*\*(.*?)\*\*", document.metadata.get('lc_id', ''))
                return match.group(1) if match else None
            # Aplica la funciÃ³n de extracciÃ³n de clave a cada documento adicional
            claves = [extraer_clave(doc) for doc in additional_documents]
            # Filtra valores no nulos y excluye los que contienen "(Ble)"
            claves = [clave for clave in claves if clave is not None]
            # Extrae la base de cada clave antes de cualquier espacio o parÃ©ntesis
            base_claves = [clave.split()[0] for clave in claves]
            # Cuenta la frecuencia de cada base
            from collections import Counter
            contador_claves = Counter(base_claves)
            # Ordena las claves Ãºnicas segÃºn su frecuencia de apariciÃ³n en orden descendente
            claves_ordenadas = [clave for clave, _ in contador_claves.most_common()]
            # Convierte la lista ordenada de claves en un string para la consulta
            claves_string = ', '.join(claves_ordenadas)
            #print(f"Claves string: {claves_string}")
            if claves_string.strip():
                additional_doc_temarios = self.retriever(namespace="Cursos").get_relevant_documents(query=claves_string)
            # Inserta los documentos adicionales al principio
                documents = additional_doc_temarios + documents
            documents_cursos = documents
            # Combina los resultados de ambos `namespace`
            documents.extend(additional_documents)  
            #print(f"Claves extraÃ­das: {claves_ordenadas}")# Debugging

        if not documents:
            return "Disculpa, no tengo la información que pides" 

        #print(f"documents1234{documents}")

       #2.) Filter and sort the documents in 3 categories
        direct_courses=[doc for doc in documents if self.safe_int_conversion(doc.metadata.get('subcontratado',0))==0]
        subcontracted_courses=[doc for doc in documents if self.safe_int_conversion(doc.metadata.get('subcontratado',0))==1]
      

        # Ordenar cursos por complejidad
        sorted_direct_courses = ordenar_cursos(direct_courses)
        sorted_subcontracted_courses = ordenar_cursos(subcontracted_courses)
        #sorted_enproceso_courses = sorted(in_process_courses, key=lambda doc:(get_complejidad_numeric(self.extract_data_from_lc_id(doc.metadata.get('lc_id', 'NA')).get('complejidad_id', 'NA')), get_entrega_num(self.extract_data_from_lc_id(doc.metadata.get('lc_id', 'NA')).get('entrega', 'NA'))))
        
        mostrar_cursos_ordenados(sorted_direct_courses + sorted_subcontracted_courses)
       
        #3.) Format the response
        formatted_documents = []
        # Procesar cursos directos
        formatted_documents.append("\n**Cursos Disponibles:**")
        labs_data = {}
        examenes_data = {}
        for doc in sorted_direct_courses:
            if 'tokens' in doc.metadata:
                try:
                    data = self.extract_data_from_lc_id(doc.metadata.get('lc_id', ''), namespace="Cursos")

                    # Buscar datos de Labs y Exámenes relacionados
                    labs_data = next(
                        (self.extract_data_from_lc_id(d.metadata.get('lc_id', ''), namespace="Laboratorios") 
                        for d in labs_documents if data.get('clave', '').strip() in d.metadata.get('lc_id', '').strip()), {}
                    )

                    examenes_data = next(
                        (self.extract_data_from_lc_id(d.metadata.get('lc_id', ''), namespace="Examenes") 
                        for d in exams_documents if data.get('clave', '').strip() in d.metadata.get('lc_id', '').strip()), {}
                    )

                    # Agregar la plantilla formateada
                    formatted_documents.append(format_response(data, labs_data, examenes_data))

                except Exception as e:
                    print(f"Error procesando curso directo: {e}")

        # Procesar cursos subcontratados
        if sorted_subcontracted_courses:
            formatted_documents.append("\n**En otras modalidades de entrega te ofrecemos:**")
            for doc in sorted_subcontracted_courses:
                if 'tokens' in doc.metadata:
                    try:
                        data = self.extract_data_from_lc_id(doc.metadata.get('lc_id', ''), namespace="Cursos")
                        # Aquí se verifica si es subcontratado
                        if "(sub)" in data.get('clave', '').lower():
                            formatted_documents.append(f"""
                            **Curso Subcontratado**  
                            - Clave: {data.get('clave', 'NA')}  
                            - Nombre: {data.get('nombre', 'NA')}  
                            - Más información: [Link])  
                            """)
                        else:

                        # seguir con el formato normal y Buscar datos de Labs y Exámenes relacionados
                            labs_data = next(
                                (self.extract_data_from_lc_id(d.metadata.get('lc_id', ''), namespace="Laboratorios") 
                                for d in labs_documents if data.get('clave', '').strip() in d.metadata.get('lc_id', '').strip()), None
                            )

                            examenes_data = next(
                                (self.extract_data_from_lc_id(d.metadata.get('lc_id', ''), namespace="Examenes") 
                                for d in exams_documents if data.get('clave', '').strip() in d.metadata.get('lc_id', '').strip()), None
                            )

                            # Agregar la plantilla formateada
                            formatted_documents.append(format_response(data, labs_data, examenes_data))

                    except Exception as e:
                        print(f"Error procesando curso subcontratado: {e}")

        document_texts=[doc for doc in formatted_documents]      
       
        history_content = self.format_history()  # Obtener el historial formateado
        system_message_content = f"{leader}{'. '.join(document_texts)}" #f"{leader1}{claves_text}{'. '.join(document_texts)}"
        system_message = SystemMessage(content=system_message_content)
          
 
        # ---------------------------------------------------------------------
        # finished with hybrid search setup
        # ---------------------------------------------------------------------
        # logging.debug("------------------------------------------------------")
        # logging.debug("rag() Retrieval Augmented Generation prompt")
        # logging.debug("Diagnostic information:")
        # logging.debug("  Retrieved %i related documents from Pinecone", len(documents))
        # logging.debug("  System messages contains %i words", len(system_message.content.split()))
        # logging.debug("  Prompt: %s", system_message.content)
        # logging.debug("---------------------------------------------------")

        response_content = ""
        try:
            print("Inicio de la función rag()")
            if self.chat.stream:
                print("Modo de transmisión activado")
                for chunk in self.chat.stream([system_message, human_message]):
                    delta_content = getattr(chunk, 'content', '')
                    if delta_content:
                        response_content += delta_content
                        try:
                            yield response_content
                        except Exception as e:
                            print(f"Error al manejar la respuesta en streaming: {e}")
                            traceback.print_exc()
                            yield 'error al transpitir la respuesta'
                #print(f"RespuestaCompleta: {response_content}")  # DEBUGGING
                if intencion == "Recomendacion" or intencion == "Cursos" or intencion == "Agnostico":
                    #claves_respuesta = re.findall(r"\*\*Clave\*\*:\s+(\S+)(?:\s*\(.*?\))?\s+-\s+\*\*Nombre\*\*:", response_content) #captura solo la base
                    #claves_respuesta = re.findall(r"\*\*Clave\*\*:\s+(\S+(?:\s*\(.*?\))?)\s+-\s+\*\*Nombre\*\*:", response_content)
                    claves_respuesta = re.findall(r"\[([A-Z0-9]+(?:\s*\(.*?\))?)\]", response_content)
                    base_claves_respuesta = list(dict.fromkeys(clave.split()[0] for clave in claves_respuesta))
                    # print(f"Claves respuesta: {claves_respuesta}")
                    # print(f"Base Claves respuesta: {base_claves_respuesta}")
                    # Asegurarse de que `system_message` sea una cadena de texto
                    if hasattr(system_message, 'content'):
                        system_message_text = system_message.content
                    else:
                        system_message_text = str(system_message)

                    informacion_cursos = {}
                    # Iterar sobre cada clave base para extraer su informaciÃ³n especÃ­fica
                    for clave in base_claves_respuesta:
                        # ExpresiÃ³n regular para encontrar el bloque de informaciÃ³n de cada clave, con coincidencia parcial
                        pattern = re.compile(rf"- Clave: {re.escape(clave)}(?:\s*\(.*?\))?(?:(?!\*\*Curso\*\*|\.\s*\n).)*", re.DOTALL)
                        matches = pattern.findall(system_message_text)
                        #print(f"Matches for clave '{clave}':\n{matches}")

                        # Filtrar los bloques en `matches` excluyendo las coincidencias exactas con `claves_respuesta`
                        filtered_matches = [
                            match for match in matches
                            if not any(full_clave == match.split('\n')[0].strip().split(': ')[-1] for full_clave in claves_respuesta)
                        ]

                        #print(f"Filtered matches for clave '{clave}':\n{filtered_matches}")

                        # Si hay coincidencias Ãºnicas despuÃ©s del filtrado, guardarlas en `informacion_cursos`
                        if filtered_matches:
                            informacion_cursos[clave] = list(set(filtered_matches))
                       
                    # Formatear e imprimir la información extraída para cada clave
                    #print("INFO DE LOS CURSOOOOOS:",informacion_cursos)
                    cursos_adicionales = ''
                    for clave, info_list in informacion_cursos.items():
                        for info in info_list:
                            if "subcontratado" in info.lower():
                                formatted_info = info
                            # Formatear la informaciónn según el estilo deseado
                                formatted_info = info.replace("- Clave:", " - **Clave**:").replace(" - Nombre:", " - **Nombre**:")
                                formatted_info = formatted_info.replace(" - Tecnologí­a:", "- **Tecnologí­a/ Lí­nea de negocio**:")
                                formatted_info = formatted_info.replace(" - Entrega:", "- **Entrega**:")
                                formatted_info = formatted_info.replace(" - Estatus del curso:", "- **Estatus del curso**:")
                                formatted_info = formatted_info.replace(" - Tipo de curso:", "- **Tipo de curso**:")
                                formatted_info = formatted_info.replace(" - Sesiones:", "- **Sesiones y Horas**:")
                                formatted_info = formatted_info.replace(" - Precio:", "- **Precio**:")                            
                                formatted_info = formatted_info.replace(" - Examen:", "- **Examen**:")
                                formatted_info = formatted_info.replace(" - Laboratorio", "- **Laboratorio**:")                                            
                                formatted_info = formatted_info.replace(" - País:", "- **País**:")
                                formatted_info = formatted_info.replace(" - Complejidad:", "- **Complejidad**:")                             
                                    
                                formatted_info = "\n".join(["   " + line.lstrip() for line in formatted_info.splitlines() if line.strip()])
                                formatted_info = formatted_info.replace(" - **Clave**:", "\n\n - **Clave**:")
                                for marker in ["- Certificación:", "- **Certificación**:"]:
                                    if marker in formatted_info:
                                        formatted_info = formatted_info.split(marker)[0]
                                        break
                                cursos_adicionales += f"{formatted_info}"
                                # Imprimir el resultado final formateado
                                #print(f" {formatted_info}")
                        if cursos_adicionales.strip():
                            response_content += f"\n\n En otras modalidades de entrega te ofrecemos: \n{cursos_adicionales}"
                            yield response_content
                    #print(f"InformaciÃ³n de cursos:\n{cursos_adicionales}")
            else:
                print("Modo de respuesta no transmitida activado")
                response = self.cached_chat_request(system_message=system_message_content, human_message=human_message)
                #print(f"Respuesta completa recibida: {response.content}")
                yield response.content
        except Exception as e:
            print(f"Error handling streaming response: {e}")
            logging.error(f"Error handling streaming response: {e}")

        # print(f"history_content: {history_content}") #DEBUGGING
        # print(f"enhanced_query: {enhanced_query}") #DEBUGGING
        # print(f"intencion: {intencion}") #DEBUGGING
        # print(f"Respuestaaaa: {response_content}") #DEBUGGING
        #print(f"system messageeeee: {system_message}")
        #print(f"ordenados: {sorted_courses}") #DEBUGGING
        #print(f"Plantillas de temarios: {additional_doc_temarios}") #DEBUGGING
        def usuario():
            global user_name
            params = st.query_params
            user_name = params.get('user_name', [None])[:]                   
        usuario()
        
        log_entry={
                "user_name":user_name or "desconocido",
                "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "question":human_message.content,
                "response":response_content
            }
        log_interaction(**log_entry)
        
       
                
       
