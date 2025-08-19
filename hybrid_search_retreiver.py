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
import requests
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
   
    # def retriever(self, namespace="Cursos", top_k=40) -> PineconeHybridSearchRetriever:
    #     return PineconeHybridSearchRetriever(
    #         embeddings=self.pinecone.openai_embeddings,
    #         sparse_encoder=self.bm25_encoder,
    #         index=self.pinecone.index,
    #         top_k=top_k,
    #         alpha=0.9,
    #         namespace=namespace
    #     )
    def retriever(self, namespace="Cursos", top_k=None, query="") -> PineconeHybridSearchRetriever:
        """
        Retriever que ajusta dinámicamente alpha y top_k según el tipo de consulta
        """
        # Analizar el tipo de consulta si se proporciona
        if query:
            query_lower = query.lower()
            #Detectar consultas que solicitan MÚLTIPLES cursos
            multiple_course_indicators = [
                "cursos", "opciones", "alternativas", "que tienes", "que hay", 
                "qué cursos", "cuáles", "todos los", "lista de", "catálogo"
            ]
            is_multiple_request= any(indicator in query_lower for indicator in multiple_course_indicators)
            # Detectar consultas específicas (necesitan más precisión textual)
            specific_indicators = [
                "clave", "precio de", "costo de", "temario de", "laboratorio de", 
                "examen de", "certificación", "duración", "horas", "sesiones"
            ]
            is_specific = any(indicator in query_lower for indicator in specific_indicators)
            
            # Detectar consultas técnicas muy específicas
            tech_specific = [
                "chatgpt", "gpt-4", "openai", "aws-", "az-", "ms-", 
                "ccna", "ccnp", "cissp", "ceh", "python", "java", "docker", "kubernetes"
            ]
            is_tech_specific = any(term in query_lower for term in tech_specific)
            
            # Detectar consultas de precios o información específica
            is_price_query = any(word in query_lower for word in ["precio", "costo", "cost", "$"])
            
            # Ajustar parámetros dinámicamente
            if is_multiple_request and not is_specific:
                # Para consultas específicas: más peso a búsqueda textual exacta
                alpha = 0.6
                top_k_dynamic = 50
            elif is_specific and not is_multiple_request:
                # Para términos técnicos específicos: balance equilibrado
                alpha = 0.3
                top_k_dynamic = 15
            elif is_tech_specific and is_multiple_request:
                alpha=.5
                top_k_dynamic=40
            elif is_price_query:
                alpha=.3
                top_k_dynamic=20
            elif is_tech_specific:
                alpha=.5
                top_k_dynamic=25
            else:
                # Para consultas generales: más peso a embeddings semánticos
                alpha = 0.7
                top_k_dynamic = 35
        else:
            # Valores por defecto balanceados
            alpha = 0.6
            top_k_dynamic = 20
        
        # Usar top_k proporcionado o el calculado dinámicamente
        final_top_k = top_k if top_k is not None else top_k_dynamic
        
        print(f"Retriever config - Alpha: {alpha}, Top_k: {final_top_k}, Query type detected")
        
        return PineconeHybridSearchRetriever(
            embeddings=self.pinecone.openai_embeddings,
            sparse_encoder=self.bm25_encoder,
            index=self.pinecone.index,
            top_k=final_top_k,
            alpha=alpha,
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
        response_content = ""
        if self.chat.stream:
            logging.debug("Chat en modo de transmisión")
            # Itera sobre los fragmentos de transmisión
            for chunk in self.chat.stream(messages):
                if isinstance(chunk, dict) and 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {}).get('content', '')
                    if delta:  # Solo agrega si el contenido no está vacío 
                        response_content += delta
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
                            "PWD=R3ad25**SC3.2025-;"
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
            complete_url = "https://sce.netec.com/" + pdf_url  

            try:
            #Verificamos si el PDF existe antes de cargarlo
                response= requests.head(complete_url)
                if response.status_code != 200:
                    print(f"⚠️ PDF no encontrado ({response.status_code}): {complete_url}")
                    continue
                #cargar PDF
                                
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
            except Exception as e:
                print(f'Error al procesar {complete_url}:{e}')
                continue
 
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

    def rag(self, human_message: Union[str, HumanMessage], conversation_history=None):
        """
        Retrieval Augmented Generation prompt simplificado.
        Toda la información de laboratorios y exámenes ya está en el namespace 'Cursos'
        """
        # Función temporal para debugging - agregar al método rag() antes de format_response()

        def debug_curso_data(data, doc_metadata=None):
            """
            Función temporal para debugging de datos de curso
            """
            print("="*50)
            print("DEBUG - DATOS DEL CURSO")
            print("="*50)
            
            if isinstance(data, dict):
                print(f"Clave: {data.get('clave', 'NO_FOUND')}")
                print(f"Nombre: {data.get('nombre', 'NO_FOUND')}")
                print(f"Subcontratado (campo): {data.get('subcontratado', 'NO_FOUND')} - Tipo: {type(data.get('subcontratado'))}")
                
                # Verificar patrones en la clave
                clave = data.get('clave', '')
                patrones_encontrados = []
                patrones_sub = ['(sub)', '(SUB)', '(Sub)', '(sub-ext)', '(SUB-EXT)', '(Sub-Ext)']
                
                for patron in patrones_sub:
                    if patron.lower() in clave.lower():
                        patrones_encontrados.append(patron)
                
                print(f"Patrones Sub encontrados en clave: {patrones_encontrados}")
                print(f"Todos los campos disponibles: {list(data.keys())}")
            
            if doc_metadata:
                print("\nDEBUG - METADATA DEL DOCUMENTO")
                print(f"Metadata subcontratado: {doc_metadata.get('subcontratado', 'NO_FOUND')}")
                print(f"Metadata lc_id: {doc_metadata.get('lc_id', 'NO_FOUND')}")
            
            print("="*50)

        try:                
            def format_response(data):
                """
                Función que genera respuestas amigables con formato limpio (SIN bloques de código)
                Con plantilla especial para cursos subcontratados
                """
                if not isinstance(data, dict):
                    print('ERROR: data no es un diccionario')
                    return "Error: Datos de curso no válidos"
                    # DEBUGGING DETALLADO
                clave_curso = data.get('clave', '')
                subcontratado_field = data.get('subcontratado', 'No')
                
                print(f"\n--- FORMAT_RESPONSE DEBUG ---")
                print(f"Clave: {clave_curso}")
                print(f"Subcontratado field: '{subcontratado_field}' (tipo: {type(subcontratado_field)})")
                
                # Verificación por clave
                patrones_sub = ['(sub)', '(SUB)', '(Sub)', '(sub-ext)', '(SUB-EXT)', '(Sub-Ext)']
                clave_indica_sub = any(patron.lower() in clave_curso.lower() for patron in patrones_sub)
                print(f"Clave indica sub: {clave_indica_sub}")
                
                # Verificación por campo
                if isinstance(subcontratado_field, str):
                    subcontratado_normalized = subcontratado_field.strip().lower()
                else:
                    subcontratado_normalized = str(subcontratado_field).strip().lower()
                
                valores_subcontratado = ['si', 'sí', 'yes', '1', 'true', 'verdadero']
                campo_indica_sub = (
                    subcontratado_normalized in valores_subcontratado or
                    subcontratado_field in [1, True, '1'] or
                    (isinstance(subcontratado_field, (int, bool)) and subcontratado_field)
                )
                print(f"Campo indica sub: {campo_indica_sub}")
                
                is_subcontratado = campo_indica_sub or clave_indica_sub
                print(f"DECISION FINAL - Es subcontratado: {is_subcontratado}")
                print(f"--- END FORMAT_RESPONSE DEBUG ---\n")
                
                if is_subcontratado:
                    print("RETORNANDO PLANTILLA SUBCONTRATADO")
                    return f"""**Curso Subcontratado**
            - **Clave**: {data.get('clave', 'NA')}
            - **Nombre**: {data.get('nombre', 'NA')}
            - **Más información**: [Subcontrataciones](https://netec.sharepoint.com/Subcontrataciones/subcontratacioneslatam/SitePages/Inicio.aspx)"""
                
                print("RETORNANDO PLANTILLA COMPLETA")
                # Si no es subcontratado, continuar con la plantilla completa
                
                # Procesar link del temario
                link_temario = data.get('link_temario', 'Temario no encontrado')
                if link_temario and link_temario != 'Temario no encontrado':
                    if link_temario.startswith("https://"):
                        link_temario = f"[Link al temario]({link_temario.replace(' ', '%20')})"
                    else:
                        link_temario = 'Temario no encontrado'
                else:
                    link_temario = 'Temario no encontrado'
                
                # Procesar información de laboratorios usando datos del curso principal
                clave_examen = data.get('clave_Examen', 'NA')
                nombre_examen = data.get('nombre_examen', 'NA')
                tipo_elemento = data.get('tipo_elemento', 'NA')
                costo = data.get('costo', 'NA')
                
                # Determinar si tiene laboratorio basado en los datos del curso
                if (tipo_elemento == 'Equipo' and 
                    nombre_examen and nombre_examen.lower().startswith('laboratorio')):
                    labs_info = f"Sí\n  - Nombre: {clave_examen} - Costo: {costo}"
                else:
                    labs_info = "No lleva laboratorio"
                
                # Determinar si tiene examen de certificación basado en los datos del curso
                if (tipo_elemento == 'Examen' and 
                    nombre_examen == 'certificación'):
                    examenes_info = f"Sí\n  - Nombre: {clave_examen} - Costo: {costo}"
                else:
                    examenes_info = "No tiene certificación asociada"
                
                # Determinar disponibilidad
                if data.get('disponible', 'NA') == 'Si' and data.get('estatus_curso', '') == 'Liberado':
                    disponibilidad = "Habilitado"
                elif data.get('disponible', 'NA') == 'No' or data.get('estatus_curso', '') != 'Liberado':
                    disponibilidad = "En habilitación"
                else:
                    disponibilidad = "NA"
                
                # PLANTILLA SIN BLOQUES DE CÓDIGO - FORMATO LIMPIO
                template = f"""**Curso**
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
                
                return template
            if not isinstance(human_message, HumanMessage):
                logging.debug("Converting human_message to HumanMessage")
                human_message = HumanMessage(content=str(human_message))

            self.add_to_history(human_message)
            intencion = self.clasificar_intencion_con_gpt(human_message.content)

            if conversation_history:
                context = " ".join([msg.content for msg in conversation_history[-5:]])
            else:
                context = ""

            def ordenar_cursos(cursos):
                """
                Ordena los cursos con manejo robusto de errores
                """
                if not cursos:
                    return []
                    
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
                    try:
                        if not hasattr(curso, 'metadata') or not isinstance(curso.metadata, dict):
                            return (1, 6, 3, 1)  # Valores por defecto
                            
                        disponible = safe_get_metadata(curso, 'disponible', 'No')
                        complejidad = safe_get_metadata(curso, 'complejidad_id', 'ADVANCED')
                        tipo_curso = safe_get_metadata(curso, 'tipo_curso_id', 'Programa')
                        subcontratado = safe_get_metadata(curso, 'subcontratado', 'No')
                        
                        return (
                            0 if disponible in ['Sí', 'Si', 'Yes', '1'] else 1,
                            complejidad_map.get(complejidad, 6),
                            tipo_curso_map.get(tipo_curso, 3),
                            0 if subcontratado in ['No', 'False', '0'] else 1
                        )
                    except Exception as e:
                        print(f"Error en ordenamiento de curso: {e}")
                        return (1, 6, 3, 1)  # Valores por defecto

                try:
                    return sorted(cursos, key=clave_ordenamiento)
                except Exception as e:
                    print(f"Error general en ordenar_cursos: {e}")
                    return cursos  # Retorna sin ordenar si hay error            

            # Resto de funciones auxiliares permanecen igual...
            def translate_with_mymemory(text):
                url = "https://api.mymemory.translated.net/get"
                params = {
                    'q': text,
                    'langpair': 'es|en'  # Traducción de español a inglés
                }

                response = requests.get(url, params=params)
                result = response.json()
                translated_text = result['responseData']['translatedText']
                return translated_text
            
            def clean_query(query):
                # Eliminar símbolos de puntuación usando una expresión regular
                query = re.sub(r'[¿?¡!]', '', query)  # Elimina los símbolos especificados
                # Lista de palabras que no aportan valor a la búsqueda
                stop_words = [
                    "agnóstico", "agnostico", "tenemos", "tienes", "tiene", "de", "a", "los", 
                    "hay", "algún", "alguna", "y", "para", "en", "el", "catalogo", "catalogo?", 
                    "cubran", "cubra", "puedo", "ofrecer", "cliente", "quiere", "conocer", 
                    "principales", "principal", "elemento", "elementos", "un", "digital"
                ]
                words=query.lower().split()
                clened_words=[]
                for word in words:
                    if word and word.strip() and word not in stop_words:
                        clened_words.append(word.strip())
                # Divide la consulta en palabras y elimina las stop words
                cleaned_queryes = " ".join(clened_words)
                if not cleaned_queryes.strip():
                    return query.strip()
                try:
                    cleaned_queryen= translate_with_mymemory(cleaned_queryes)
                except Exception as e:
                    print(f'Error en traducción:{e}')
                    cleaned_queryen=cleaned_queryes
                
                #devolver la consulta combinada
                if cleaned_queryes.strip()==cleaned_queryen.strip():
                    return cleaned_queryes
                else:
                    return f'{cleaned_queryes}{cleaned_queryen}'
            def get_unified_leader_prompt(intencion):
                """
                Función que genera prompts unificados manteniendo las instrucciones específicas de cada intención 
                pero garantizando el uso de la plantilla obligatorio.
                """
                # PLANTILLA OBLIGATORIA - NUNCA CAMBIAR
                TEMPLATE_OBLIGATORIA = """
            **FORMATO OBLIGATORIO PARA CADA CURSO - USAR EXACTAMENTE COMO SE MUESTRA:**
            
            Para cada curso que no sea subcontratado, debes seguir esta estructura OBLIGATORIA:
            1. Descripción detallada del curso basada en el temario (mínimo 150 palabras)
            2. Información estructurada del curso usando la plantilla

            **Plantilla del curso**
            - **Clave**: [valor]
            - **Nombre**: [valor] V[versión]
            - **Tecnología/ Línea de negocio**: [valor] / [valor]
            - **Entrega**: [valor]
            - **Estatus del curso**: [valor]
            - **Tipo de curso**: [valor]
            - **Sesiones y Horas**: [valor] / [valor]
            - **Precio**: [valor] [moneda]
            - **Examen**: [valor]
            - **Laboratorio**: [valor]
            - **País**: [valor]
            - **Complejidad**: [valor]
            - **Link al temario**: [valor]

            REGLAS CRÍTICAS:
            - SIEMPRE responde de forma amable y conversacional ANTES de mostrar la información del curso
            - USA EXACTAMENTE esta plantilla para cada curso que no sea subcontratado (SIN bloques de código ``` ```)
            - NO pongas la plantilla dentro de bloques de código
            - NO uses títulos como "RESUMEN DEL TEMARIO" o "PLANTILLA DEL CURSO"
            - Formatea la respuesta de manera limpia y legible
            - NO cambies el orden de los campos
            - NO omitas ningún campo
            - NO modifiques los nombres de los campos
            - Sé amable y profesional en tu respuesta"""

                # BASE COMÚN
                base_prompt = f"""Eres Max, una asistente amable y experta en cursos de NETEC que ayuda a encontrar la mejor capacitación tecnológica.

            Tu personalidad:
            - Siempre amable, profesional y servicial
            - Usas un tono conversacional y cálido
            - Explicas de manera clara y comprensible
            - Siempre buscas ayudar al máximo

            {TEMPLATE_OBLIGATORIA}
            """

                # INSTRUCCIONES ESPECÍFICAS POR INTENCIÓN
                specific_instructions = {
                    "Certificaciones": """
            **Instrucciones para CERTIFICACIONES**:
            - Si la consulta incluye términos como "certificación" o "certificaciones", muestra SÓLO la lista de certificaciones disponibles
            - NO incluyas cursos donde el campo "Certificación" sea "Ninguna" o esté vacío
            - Cada certificación se menciona una sola vez (no repetir si varios cursos la comparten)
            - Si preguntan sobre disponibilidad de una certificación, responde y sugiere cursos asociados
            - NO incluyas cursos adicionales
            - DESPUÉS de listar certificaciones, si hay cursos asociados, muestra cada uno con la plantilla completa
            - IMPORTANTE: Examen de certificación no es igual que examen de curso
            """,
                    
                    "Precio": """
            **Instrucciones para PRECIO**:
            - SIEMPRE comienza de forma amable: "¡Por supuesto! Te comparto la información del precio que solicitas:"
            - Si la consulta incluye "precio" o "costo" y menciona un curso específico: proporciona sólo el costo de lo que te piden. Por ejemplo : "¡Por supuesto! Te comparto la información del precio que solicitas: Curso PYTHON (Digital) tiene un costo de  699.0 USD"
            - Si es una consulta general de precios: "Aquí tienes los precios de los cursos que encontré:" seguido de la lista de costos asociado cada uno a su clave
            - Si menciona examen o laboratorio específico: muestra la información dentro de la plantilla completa del curso
            - Mantén un tono amable y profesional
            """,
                    
                    "General": """
            **Instrucciones para CONSULTAS GENERALES**:
            - SIEMPRE comienza con una respuesta amable y conversacional
            - Si el usuario saluda, salúdalo calurosamente y ofrece ayuda
            - Para temas/tecnologías específicas: "¡De acuerdo! He encontrado algunos cursos relacionados con tu consulta que podrían interesarte:"
            - Para recomendaciones a clientes: "Te comparto las opciones disponibles para este tema:"
            - Si hay cursos subcontratados: usa el formato correspondiente y añade al final "Te recomiendo ponerte en contacto con un Ing. preventa para más información sobre esta modalidad"
            - Para servicios específicos: añade al final "Te sugiero conversar con un Ing. preventa para profundizar en los detalles técnicos"

            **FORMATO OBLIGATORIO PARA CADA CURSO**:
            1. Respuesta conversacional amable
            2. Para cada curso: **Nombre del curso [Clave]** (en negritas)
            3. Resumen humanizado usando información de **Temarios** (mínimo 150 palabras por curso)
            4. Inmediatamente después, la plantilla completa del curso SIN bloques de código
            5. SIEMPRE sé amable y profesional
            """,
                    
                    "Agnostico": """
            **Instrucciones para CURSOS AGNÓSTICOS**:
            - Si incluye "agnóstico/agnósticos": muestra ÚNICAMENTE cursos que contengan "Comptia", "CCNA", "APMG" o "GKI"
            - NO muestres cursos con "Microsoft", "AWS", "Cisco", "ECCouncil", o "Palo Alto"
            - Haz resúmenes usando información de **Temarios** (mínimo 150 palabras por curso)
            - Formato: nombre del curso y clave en negritas, seguido del resumen, luego la plantilla completa del curso
            - Si no encuentras cursos agnósticos: "Disculpa, no tengo esta información. Favor de ponerte en contacto con un Ing. Preventa."
            - SIEMPRE usa la plantilla obligatoria para cada curso mostrado
            """,
                    
                    "Temarios": """
            **Instrucciones para TEMARIOS**:
            - Responde de manera humanizada la consulta del usuario
            - Si piden temario de un curso: proporciona el enlace del temario, añade una breve descripción y la plantilla completa del curso
            - Ejemplo: "Puedes encontrar el temario del curso AZ-900T00 en: [enlace]"
            - USA la información de **Listado de temarios**
            - SIEMPRE incluye la plantilla completa del curso junto con el enlace al temario
            """,
                    
                    "Recomendacion": """
            **Instrucciones para RECOMENDACIONES**:
            - Responde de manera humanizada usando información de **Listado de cursos y temarios**
            - ORDEN: Primero cursos habilitados, luego cursos en habilitación y al final cursos subcontratados
            - CRITERIO DE ORDENACIÓN:
            * Por complejidad: CORE(1), FOUNDATIONAL(2), BASIC(3), INTERMEDIATE(4), SPECIALIZED(5), ADVANCED(6)
            * Por tipo de curso: Intensivo(1), Digital(2), Programa(3)
            - Para consultas sobre "versión" específica: responde sólo con esa información usando la plantilla completa
            - Si existen varios cursos con clave similar: incluye TODOS usando la plantilla completa (ej: CCNA, CCNA Digital, CCNA Digital-CLC)
            - Para temas desconocidos: "Disculpa, no tengo esta información. Favor de contactar a un Ing. Preventa"
            - Haz resúmenes usando **Temarios** (mínimo 150 palabras por curso)
            
            """,
                    
                    "Laboratorio": """
            **Instrucciones para LABORATORIOS**:
            - Cuando pregunten por laboratorio de un curso deberás mencionar si este existe, incluye su clave y costo. Si no tiene, mencionalo de forma amable
            - La información del laboratorio debe aparecer en el campo **Laboratorio** de la plantilla
            
            """,
                    
                    "Examenes": """
            **Instrucciones para EXÁMENES**:
            - Cuando pregunten por examen de un curso deberás mencionar si este existe , incluye su clave y costo. Si no tiene, mencionalo de forma amable
            - La información del examen debe aparecer en el campo **Examen** de la plantilla
            - IMPORTANTE: Examen de certificación no es igual que examen de curso
            """,
                    
                    "Cursos": """
            **Instrucciones para CURSOS**:
            - Para consultas generales sobre cursos: usa la plantilla completa
            - Si mencionan un curso específico: muestra la plantilla completa de ese curso
            - Para comparaciones: muestra la plantilla completa de cada curso mencionado
            - Haz resúmenes usando información de **Temarios** cuando sea apropiado
            - SIEMPRE usa la plantilla obligatoria para cada curso mostrado
            """,
                    
                    "No": """
            **Instrucciones para CONSULTAS NO CLARAS**:
            - Si el usuario saluda: salúdalo y ofrece ayuda amablemente
            - Para consultas no claras: "¡Hola! Parece que tu consulta no está clara o no está relacionada con nuestros temas. ¿Podrías darme más detalles?"
            - Para información no disponible: "Disculpa, no tengo esa información. Favor de ponerte en contacto con un Ing. Preventa"
            - Si eventualmente muestras algún curso, usa la plantilla obligatoria
            """
                }
                
                # Obtener instrucciones específicas
                specific_instruction = specific_instructions.get(intencion, specific_instructions["No"])                
                # Combinar todo
                full_prompt = f"""{base_prompt}

            {specific_instruction}

            **Listado de cursos y temarios:**"""                
                return full_prompt

            namespace_map={
                "Temarios": "Temarios"
            }
            namespace= namespace_map.get(intencion, "Cursos")
            leader=get_unified_leader_prompt(intencion)    

            enhanced_query = f"{clean_query(human_message.content)}"  
            if not enhanced_query.strip():
                print("Consulta vacía, se omite la búsqueda en Pinecone.")
                return "Disculpa, no puedo procesar una consulta vacía. Por favor, intenta nuevamente."

            # Realizar la búsqueda SOLO en el namespace principal
            try:
                #documents = self.retriever(namespace=namespace).get_relevant_documents(query=enhanced_query)
                documents = self.retriever(namespace=namespace, query=enhanced_query).get_relevant_documents(query=enhanced_query)
                print(f"Documentos recuperados en {namespace}: {len(documents)}")
                
                print("="*60)
                print("DEBUGGING CLASIFICACIÓN DE CURSOS")
                print("="*60)

                for i, doc in enumerate(documents):
                    try:
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                            lc_id = doc.metadata.get('lc_id', 'Sin lc_id')
                            subcontratado_meta = doc.metadata.get('subcontratado', 'Sin dato')
                        else:
                            lc_id="No metadata"
                            subcontratado_meta="No metadata"
                            # Extraer datos para ver la clave
                            try:
                                data = self.extract_data_from_lc_id(lc_id, namespace="Cursos")
                                if isinstance(data,dict):
                                    clave = data.get('clave', 'Sin clave')
                                    subcontratado_data = data.get('subcontratado', 'Sin dato')
                                    
                                    print(f"\nDOC {i+1}:")
                                    print(f"  Clave: {clave}")
                                    print(f"  Subcontratado (metadata): {subcontratado_meta}")
                                    print(f"  Subcontratado (data): {subcontratado_data}")
                                    print(f"  Tiene (Sub) en clave: {'(sub)' in clave.lower() or '(SUB)' in clave}")
                                    
                                    tiene_sub_patron = any(patron in clave.upper() for patron in [' (SUB)', ' (SUB-EXT)',' (Sub) (Digital)',' (Sub-Ext)',' (sub)',' (Sub)'])
                                    print(f"  Tiene (Sub) en clave: {tiene_sub_patron}")
                                else:
                                    print(f"\nDOC {i+1}: Error - extract_data_from_lc_id no devolvió diccionario")
                                    
                            except Exception as e:
                                print(f"\nDOC {i+1}: Error al extraer datos - {e}")
                    except Exception as e:
                                print(f"\nDOC {i+1}: Error al extraer datos - {e}")

                print("="*60)


                def safe_get_metadata(doc, key, default=None):
                    """Función auxiliar mejorada para acceder de forma segura a los metadatos"""
                    try:
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                            value = doc.metadata.get(key, default)
                            if value is None:
                                return default
                            return value
                        elif hasattr(doc, key):
                            return getattr(doc, key, default)
                        else:
                            print(f"Documento sin metadata válido para key '{key}'")
                            return default
                    except Exception as e:
                        print(f"Error accediendo a metadata '{key}': {e}")
                        return default
                
                
                def safe_int_conversion(value, default=0):
                    """Conversión segura a entero"""
                    if value is None:
                        return default
                    if isinstance(value, (str, int, float)):
                        try:
                            if isinstance(value, str):
                                value_lower = value.lower().strip()
                                if value_lower in ['si', 'sí', 'yes', '1', 'true']:
                                    return 1
                                elif value_lower in ['no', 'false', '0']:
                                    return 0
                            return int(float(value))
                        except (ValueError, TypeError):
                            return default
                    return default

                try:
                    documents = sorted(documents, key=lambda doc: safe_get_metadata(doc, "orden", 0))
                    print(f"Documentos recuperados en {namespace}: {len(documents)}")
                    
                    for doc in documents:
                        lc_id = safe_get_metadata(doc, 'lc_id', 'No ID')
                        print(f"Documento en {namespace}: {lc_id}")
                        
                except Exception as e:
                    print(f"Error al ordenar documentos: {e}")
                    print(f"Documentos recuperados en {namespace} (sin ordenar): {len(documents)}")

            except Exception as e:
                print(f"Error al realizar la búsqueda en Pinecone: {e}")
                logging.error(f"Error al realizar la búsqueda en Pinecone: {e}")
                return "Disculpa, hubo un problema al realizar la búsqueda. Intenta nuevamente."
            
            # ELIMINAMOS TODA LA LÓGICA DE BÚSQUEDAS ADICIONALES EN LABORATORIOS Y EXAMENES
            # Ya no necesitamos estas búsquedas separadas porque toda la info está en "Cursos"
            
            # Para búsquedas de temarios, mantenemos la lógica existente
            if intencion == "Recomendacion" or intencion == "Cursos" or intencion == "Agnostico": 
                additional_namespace = "Temarios" if namespace == "Cursos" else "Cursos"
                additional_documents = self.retriever(namespace=additional_namespace).get_relevant_documents(query=enhanced_query)
                
                # Lógica de extracción de claves permanece igual...
                def extraer_clave(document):
                    match = re.search(r"Clave:\s*\*\*(.*?)\*\*", document.metadata.get('lc_id', ''))
                    return match.group(1) if match else None

                claves = [extraer_clave(doc) for doc in additional_documents]
                claves = [clave for clave in claves if clave is not None]
                base_claves = [clave.split()[0] for clave in claves]
                
                from collections import Counter
                contador_claves = Counter(base_claves)
                claves_ordenadas = [clave for clave, _ in contador_claves.most_common()]
                claves_string = ', '.join(claves_ordenadas)
                
                if claves_string.strip():
                    additional_doc_temarios = self.retriever(namespace="Cursos").get_relevant_documents(query=claves_string)
                    documents = additional_doc_temarios + documents
                
                documents_cursos = documents
                documents.extend(additional_documents)  

            if not documents:
                return "Disculpa, no tengo la información que pides" 

            # 2.) Filter and sort the documents in 3 categories
            try:
                direct_courses = []
                subcontracted_courses = []
                
                for doc in documents:
                    try:
                        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                            print(f"Documento sin metadata válido: {doc}")
                            continue
                            
                        subcontratado_value = safe_get_metadata(doc, 'subcontratado', '0')
                        subcontratado_int = safe_int_conversion(subcontratado_value, 0)
                        
                        if subcontratado_int == 1:
                            subcontracted_courses.append(doc)
                        else:
                            direct_courses.append(doc)
                            
                    except Exception as e:
                        print(f"Error procesando documento individual: {e}")
                        direct_courses.append(doc)
                        continue

                try:
                    sorted_direct_courses = ordenar_cursos(direct_courses)
                except Exception as e:
                    print(f"Error ordenando cursos directos: {e}")
                    sorted_direct_courses = direct_courses
                    
                try:
                    sorted_subcontracted_courses = ordenar_cursos(subcontracted_courses)
                except Exception as e:
                    print(f"Error ordenando cursos subcontratados: {e}")
                    sorted_subcontracted_courses = subcontracted_courses

            except Exception as e:
                print(f"Error general en filtrado y ordenamiento: {e}")
                sorted_direct_courses = documents
                sorted_subcontracted_courses = []

            # 3.) Format the response - SIMPLIFICADO
            formatted_documents = []
            
            # Procesar cursos directos
            if sorted_direct_courses:
                formatted_documents.append("\n**Cursos Disponibles:**")
                for doc in sorted_direct_courses:
                    try:
                        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                            print(f"Saltando documento sin metadata: {doc}")
                            continue
                            
                        lc_id = safe_get_metadata(doc, 'lc_id', '')
                        
                        if not lc_id or lc_id.strip() == '':
                            print("Documento sin lc_id válido")
                            continue
                            
                        data = self.extract_data_from_lc_id(lc_id, namespace="Cursos")
                        debug_curso_data(data,doc.metadata)
                        
                        if not isinstance(data, dict):
                            print(f"Error: extract_data_from_lc_id no devolvió un diccionario válido para {lc_id}")
                            continue

                        # SIMPLIFICADO: Ya no buscamos labs_data y examenes_data por separado
                        # Toda la información está en 'data'
                        formatted_response = format_response(data)
                        formatted_documents.append(formatted_response)
                        
                    except Exception as e:
                        print(f"Error procesando curso directo: {e}")
                        print(f"Documento problemático: {doc}")
                        continue

            # Procesar cursos subcontratados - SIMPLIFICADO
            if sorted_subcontracted_courses:
                print("\n" + "="*50)
                print("PROCESANDO CURSOS SUBCONTRATADOS")
                print("="*50)
                
                formatted_documents.append("\n**En otras modalidades de entrega te ofrecemos:**")
                for i, doc in enumerate(sorted_subcontracted_courses):
                    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                        try:
                            lc_id = doc.metadata.get('lc_id', '')
                            print(f"\nSUBCONTRATADO {i+1}:")
                            print(f"  lc_id: {lc_id}")
                            
                            data = self.extract_data_from_lc_id(lc_id, namespace="Cursos")
                            print(f"  Data extraída: {isinstance(data, dict)}")
                            
                            if isinstance(data, dict):
                                print(f"  Clave: {data.get('clave', 'NA')}")
                                print(f"  Subcontratado: {data.get('subcontratado', 'NA')}")
                                print(f"  Tipo subcontratado: {type(data.get('subcontratado'))}")
                            
                            formatted_response = format_response(data)
                            print(f"  Respuesta generada (primeros 100 chars): {formatted_response[:100]}...")
                            formatted_documents.append(formatted_response)

                        except Exception as e:
                            print(f"Error procesando curso subcontratado {i+1}: {e}")
            # Armado seguro del system_message_content
            system_message_content = f"{leader}{'. '.join(str(doc) for doc in formatted_documents)}"
            system_message = SystemMessage(content=system_message_content)

            # Resto del código de streaming permanece igual...
            response_content = ""
            try:
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
                                yield f'Error al transmitir la respuesta: {str(e)}'

                    # Resto de la lógica de procesamiento post-respuesta...
                    if intencion == "Recomendacion" or intencion == "Cursos" or intencion == "Agnostico":
                        # Esta lógica permanece igual...
                        pass
                        
                else:
                    print("Modo de respuesta no transmitida activado")
                    response = self.cached_chat_request(system_message=system_message_content, human_message=human_message)
                    yield response.content
                    
            except Exception as e:
                print(f"Error general en streaming: {e}")
                logging.error(f"Error general en streaming: {e}")
                yield f'Error durante la transmisión de respuesta: {str(e)}'

            # Logging permanece igual...
            def usuario():
                global user_name
                params = st.query_params
                user_name = params.get('user_name', [None])[:]                   
            usuario()
            
            log_entry = {
                    "user_name": user_name or "desconocido",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "question": human_message.content,
                    "response": response_content
                }
            log_interaction(**log_entry)

        except Exception as e:
            print(f"❌ ERROR CAPTURADO EN RAG: {e}")
            print(f"Tipo: {type(e)}")
            import traceback
            traceback.print_exc()
            return f"Error durante la transmisión de respuesta: {str(e)}"
