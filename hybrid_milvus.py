#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Search Retriever adaptado para Milvus. Combina:
    - OpenAI prompting y ChatModel
    - PromptingWrapper
    - Vector embedding con Milvus
    - Hybrid Retriever para combinar embeddings vectoriales con búsqueda de texto
    - Identificación de país del usuario para filtrado

Migrado de Pinecone a Milvus local manteniendo toda la funcionalidad original
"""

# Imports generales
import traceback
import requests
import logging
import datetime
from typing import Union, List, Dict, Any
import pyodbc
import json
import urllib.parse
import uuid
import tempfile
import re
import os
from dotenv import load_dotenv
from collections import Counter
from pydantic import BaseModel

# Milvus integration
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from langchain_community.cache import InMemoryCache
import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# BM25 para búsqueda híbrida
try:
    from pinecone_text.sparse import BM25Encoder
except ImportError:
    try:
        from rank_bm25 import BM25Okapi
        print("Using rank_bm25 as BM25 implementation")
        BM25Encoder = None
    except ImportError:
        print("Warning: No BM25 implementation found")
        BM25Encoder = None

# Configuración y logging
from .logger import log_interaction
from .conf import settings
os.environ['PYTHONUTF8'] = '1'

class CustomPromptModel(BaseModel):
    prompt_template: PromptTemplate

    class Config:
        arbitrary_types_allowed = True

# === Helpers de usuario y país (NUEVOS) ===
def usuario_desde_query_params() -> str | None:
    """
    Obtiene el nombre de usuario desde la URL de Streamlit (?user_name=Nombre Apellido).
    Si no viene, intenta usar st.session_state.get('user_name').
    """
    try:
        # Streamlit moderno
        if hasattr(st, 'query_params'):
            user_name = st.query_params.get("user_name", None)
            if user_name:
                return user_name if isinstance(user_name, str) else str(user_name)
        
        # Fallback a session_state
        return st.session_state.get("user_name", None)
    except Exception as e:
        print(f"Error obteniendo usuario: {e}")
        return None

def get_user_country(user_name: str) -> str | None:
    """
    Consulta el país del usuario en SQL: users.pais_id -> paises.nombre
    """
    if not user_name:
        return None

    # 1) Partir el nombre completo
    names = user_name.strip().split(" ", 1)
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ""

    # 2) Conexión a SQL (usa tu cadena actual) - CORREGIDA
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;"
        "DATABASE=netec_prod;"
        "UID=netec_read;"
        "PWD=R3ad25**SC3.2025-;"
        "TrustServerCertificate=yes;"
    )
    conn = None
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        sql = """
        SELECT p.nombre AS pais
        FROM [netec_prod].[dbo].[users] u
        INNER JOIN [netec_prod].[dbo].[paises] p
            ON u.pais_id = p.id
        WHERE u.first_name = ? AND u.last_name = ? AND u.email LIKE '%netec%';
        """
        cursor.execute(sql, first_name, last_name)
        row = cursor.fetchone()
        country = row.pais if row else None
        print(f"País encontrado para {user_name}: {country}")
        return country
    except Exception as e:
        print(f"[get_user_country] Error obteniendo país (continuando sin filtro): {e}")
        return None  # Continuar sin filtro de país si hay error
    finally:
        try:
            if conn: conn.close()
        except: 
            pass

def download_pdf(url):
    """Descargar PDF desde URL"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error al descargar el PDF")

def load_full_text(url):
    """Cargar texto completo desde PDF"""
    pdf_content = download_pdf(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_content)
        temp_pdf_path = temp_pdf.name

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

class MilvusIndex:
    """Clase para manejar conexiones e índices de Milvus - reemplaza PineconeIndex"""
    
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.message_history = []
        self._openai_embeddings = None
        self._text_splitter = None
        self._vector_store = None
        self.connect()
    
    def connect(self):
        """Conectar a Milvus"""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            print(f"Conectado a Milvus en {self.host}:{self.port}")
        except Exception as e:
            print(f"Error conectando a Milvus: {e}")
            raise
    
    @property
    def openai_embeddings(self):
        """OpenAI embeddings lazy property"""
        if self._openai_embeddings is None:
            self._openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key.get_secret_value()
            )
        return self._openai_embeddings
    
    @property
    def text_splitter(self):
        """Text splitter lazy property"""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        return self._text_splitter
    
    @property 
    def vector_store(self):
        """Vector store para compatibilidad con código original"""
        if self._vector_store is None:
            self._vector_store = MilvusVectorStore()
        return self._vector_store
    
    def get_collection(self, collection_name: str):
        """Obtener colección de Milvus"""
        try:
            collection = Collection(collection_name)
            collection.load()
            return collection
        except Exception as e:
            print(f"Error obteniendo colección {collection_name}: {e}")
            return None
    
    def tokenize(self, text: str, prioritized_columns: List[str] = None):
        """Tokenizar texto - método de compatibilidad"""
        return text.split()

class MilvusVectorStore:
    """Clase auxiliar para manejar operaciones de vector store en Milvus"""
    
    def add_documents(self, documents, embeddings, namespace="cursos"):
        """Agregar documentos a Milvus - reemplaza el método de Pinecone"""
        try:
            # Mapear namespace a nombre de colección
            collection_name = namespace.lower()
            
            collection = Collection(collection_name)
            
            # Preparar datos para inserción
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Extraer campos específicos para Milvus
            pais_values = []
            pais_curso_values = []
            
            for metadata in metadatas:
                pais_values.append(metadata.get("pais", "NA"))
                pais_curso_values.append(metadata.get("pais_curso", "NA"))
            
            # Preparar entidades para inserción
            entities = [
                embeddings,     # vector embeddings
                texts,          # texto
                pais_values,    # país
                pais_curso_values  # país curso
            ]
            
            # Insertar en Milvus
            insert_result = collection.insert(entities)
            collection.flush()
            
            print(f"{len(documents)} documentos insertados en colección '{collection_name}'")
            return insert_result
            
        except Exception as e:
            print(f"Error insertando documentos en Milvus: {e}")
            raise

class MilvusHybridRetriever:
    """Retriever híbrido para Milvus que combina búsqueda vectorial y BM25"""
    
    def __init__(self, collection_name: str, embeddings, top_k: int = 20, alpha: float = 0.6, filter_expr: str = None):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.top_k = top_k
        self.alpha = alpha
        self.filter_expr = filter_expr
        
        # Obtener colección
        self.collection = Collection(collection_name)
        self.collection.load()
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Obtener documentos relevantes usando búsqueda híbrida"""
        try:
            # Búsqueda vectorial con Milvus
            return self._vector_search(query)
        except Exception as e:
            print(f"Error en búsqueda híbrida: {e}")
            return []
    
    def _vector_search(self, query: str) -> List[Document]:
        """Realizar búsqueda vectorial en Milvus"""
        try:
            # Generar embedding de la consulta
            query_vector = self.embeddings.embed_query(query)
            
            # Parámetros de búsqueda
            search_params = {
                "metric_type": "IP",  # Inner Product
                "params": {"nprobe": 10}
            }
            
            # Realizar búsqueda con filtro opcional
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=self.top_k,
                expr=self.filter_expr,  # NUEVO: Soporte para filtros de país
                output_fields=["text", "pais", "pais_curso"]
            )
            
            # Convertir resultados a formato Document
            documents = []
            for hit in results[0]:
                fields = hit.fields
                doc = Document(
                    page_content=fields.get("text", ""),
                    metadata={
                        "score": hit.score,
                        "id": hit.id,
                        "pais": fields.get("pais", ""),
                        "pais_curso": fields.get("pais_curso", ""),
                        "lc_id": fields.get("text", ""), 
                        "orden": 0,  # Orden por defecto
                        "context": fields.get("text", ""),
                        "tokens": fields.get("text", "").split()
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error en búsqueda vectorial: {e}")
            return []

class HybridSearchRetriever:
    """Hybrid Search Retriever adaptado para Milvus"""

    _chat: ChatOpenAI = None
    _b25_encoder = None
    _milvus: MilvusIndex = None
    _retriever = None

    def __init__(self):
        set_llm_cache(InMemoryCache())
        self.milvus.message_history = []
        
    def add_to_history(self, message: BaseMessage):
        self.milvus.message_history.append(message)
        
    def format_history(self):
        # Limita el historial a las últimas n interacciones
        max_interactions = 5
        return " ".join([
            f"Usuario: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
            for msg in self.milvus.message_history[-max_interactions:]
        ])
        
    def get_history(self):
        return self.milvus.message_history

    @property
    def milvus(self) -> MilvusIndex:
        """MilvusIndex lazy read-only property - reemplaza pinecone property"""
        if self._milvus is None:
            self._milvus = MilvusIndex()
        return self._milvus
    
    @property
    def chat(self) -> ChatOpenAI:
        if self._chat is None:
            self._chat = ChatOpenAI(
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
    def bm25_encoder(self):
        """BM25Encoder lazy read-only property"""
        if self._b25_encoder is None and BM25Encoder:
            try:
                self._b25_encoder = BM25Encoder().default()
            except:
                self._b25_encoder = None
        return self._b25_encoder

    def retriever(self, namespace="Cursos", top_k=None, query="", pais_usuario=None) -> MilvusHybridRetriever:
        """
        Retriever que ajusta dinámicamente alpha y top_k según el tipo de consulta
        CON SOPORTE PARA FILTRADO POR PAÍS
        Args:
            namespace: Namespace/coleccion a buscar
            top_k: Número de resultados (None para auto)
            query: Query de búsqueda
            pais_usuario: País del usuario obtenido de la BD
        """
        # Mapeo de namespaces a colecciones de Milvus
        collection_map = {
            "Cursos": "cursos",
            "Temarios": "temarios",
            "Laboratorios": "cursos",  # Todo está en cursos ahora
            "Examenes": "cursos"       # Todo está en cursos ahora
        }
        
        collection_name = collection_map.get(namespace, "cursos")
        
        # Analizar el tipo de consulta si se proporciona
        if query:
            query_lower = query.lower()
            
            # Detectar consultas que solicitan MÚLTIPLES cursos
            multiple_course_indicators = [
                "cursos", "opciones", "alternativas", "que tienes", "que hay", 
                "qué cursos", "cuáles", "todos los", "lista de", "catálogo", "que cursos"
            ]
            is_multiple_request = any(indicator in query_lower for indicator in multiple_course_indicators)
            
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
                # Para consultas múltiples: más weight a búsqueda textual exacta
                alpha = 0.6
                top_k_dynamic = 50
            elif is_specific and not is_multiple_request:
                # Para consultas específicas: más peso a búsqueda textual exacta
                alpha = 0.3
                top_k_dynamic = 15
            elif is_tech_specific and is_multiple_request:
                alpha = 0.5
                top_k_dynamic = 40
            elif is_price_query:
                alpha = 0.3
                top_k_dynamic = 20
            elif is_tech_specific:
                alpha = 0.5
                top_k_dynamic = 25
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
        
        # Filtrado por país
        filter_expr= None
        
        # Filtrar por país individual
        if pais_usuario:
            filter_expr=f'(pais_curso== "{pais_usuario}") or (pais_curso=="LATAM" and pais=="{pais_usuario}")'
            print(f"Filtro de país aplicado para usuario de {pais_usuario}: {filter_expr}")
        else:
            print("Sin filtro de país-mostrando todos los cursos disponibles")
        
        print(f"Retriever confg- Collection: {collection_name}, Alpha: {alpha}, Top_k:{final_top_k}")
            
        return MilvusHybridRetriever(
            collection_name=collection_name,
            embeddings=self.milvus.openai_embeddings,
            top_k=final_top_k,
            alpha=alpha,
            filter_expr=filter_expr  # NUEVO: Pasar filtro a Milvus
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
        
        messages = [SystemMessage(content=f"{self.format_history()}"), system_message, human_message]
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
            api_key=settings.openai_api_key.get_secret_value(),
            organization=settings.openai_api_organization
        )
        retval = llm(prompt.format(concept=concept))
        return retval

    def detect_language_from_key(self, clave):
        """
        Detecta el idioma del curso basado en la clave.
        Si contiene 'ESP' es español, si no es inglés-
        """
        if isinstance(clave, str) and 'ESP' in clave.upper():
            return "Español"
        return "Inglés"
    
    def extract_data_from_lc_id(self, lc_id_value, namespace="Cursos"):
        """Extraer datos del lc_id - adaptado para formato de Milvus"""
        print("Ejecutando extract_data_from_lc_id...")
        print(f"lc_id_value recibido: {repr(lc_id_value)}")
        
        global values
    
        if isinstance(lc_id_value, str) and '\n' in lc_id_value:
            # Parsear el texto estructurado
            data = {}
            lines = lc_id_value.split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Mapear campos según el formato esperado
                    field_mapping = {
                        'Clave': 'clave',
                        'Nombre': 'nombre',
                        'Certificación': 'certificacion',
                        'Disponible': 'disponible',
                        'Sesiones': 'sesiones',
                        'Precio': 'precio',
                        'Subcontratado': 'subcontratado',
                        'Pre-requisitos': 'pre_requisitos',
                        'Tecnología': 'tecnologia_id',
                        'Complejidad': 'complejidad_id',
                        'Moneda':'nombre_moneda',
                        'Tipo de curso': 'tipo_curso_id',
                        'Moneda de examen': 'nombre_moneda_examenes',
                        'Estatus': 'estatus_curso',
                        'Familia': 'familia_id',
                        'Horas': 'horas',
                        'Link temario': 'link_temario',
                        'Línea de negocio': 'linea_negocio_id',
                        'Versión': 'version',
                        'Entrega': 'entrega',
                        'País del curso': 'pais_curso',
                        'Clave examen': 'clave_examen',
                        'Nombre examen': 'nombre_examen',
                        'Base costo examen': 'base_costo_examen',
                        'Costo examen': 'costo_examen',
                        'Clave laboratorio': 'clave_laboratorio',
                        'Nombre laboratorio': 'nombre_laboratorio',
                        'Base costo laboratorio': 'base_costo_laboratorio',
                        'Costo laboratorio': 'costo_laboratorio',
                        'País (elemento)': 'pais',
                        'Retake':'incluye_re_take',
                        'Costo retake':'costo_re_take',
                        'Idioma': 'idioma'
                    }
                    
                    mapped_key = field_mapping.get(key)
                    if mapped_key:
                        data[mapped_key] = value
            
            # Detectar idioma si no está presente
            if 'idioma' not in data and 'clave' in data:
                data['idioma'] = self.detect_language_from_key(data['clave'])
            
            print(f"Datos extraídos de texto estructurado: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return data
        
        # Fallback a lógica de separación por ';' (para compatibilidad)
        values = lc_id_value.split(';') if isinstance(lc_id_value, str) else [str(lc_id_value)]
        print(f"Valores después de split: {values}")
        print(f"Cantidad de valores en lc_id_value: {len(values)}")

        if namespace == "Cursos":
            keys = ["clave", "nombre", "certificacion", "disponible", "sesiones", "precio",
                    "subcontratado", "pre_requisitos", "tecnologia_id", "complejidad_id","nombre_moneda",
                    "tipo_curso_id", "nombre_moneda_examenes", "estatus_curso", "familia_id", "horas",
                    "link_temario", "linea_negocio_id", "version", "entrega", "pais_curso",
                    "clave_examen", "nombre_examen", "base_costo_examen", "costo_examen",
                    "clave_laboratorio", "nombre_laboratorio", "base_costo_laboratorio",
                    "costo_laboratorio", "pais", "incluye_re_take", "costo_re_take"]
    
        values.extend(['NA'] * (len(keys) - len(values)))
        extracted_data = dict(zip(keys, values))
        
        # Detectar idioma basado en la clave
        clave_curso = extracted_data.get("clave", "").strip()
        idioma = self.detect_language_from_key(clave_curso)
        extracted_data["idioma"] = idioma
        
        print(json.dumps(extracted_data, indent=2, ensure_ascii=False))
        return extracted_data   
       
    def load_sql(self, sql=None, namespace="Cursos"):  # Cambiar parámetro por defecto
        """Load sql database - adaptado para Milvus"""
        # Conectar a la BD
        connectionString = ("DRIVER={ODBC Driver 18 for SQL Server};"
                        "SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;" 
                        "DATABASE=netec_prod;"
                        "UID=netec_read;"
                        "PWD=R3ad55**N3teC+*;"
                        "TrustServerCertificate=yes;")
        conn = pyodbc.connect(connectionString)
        cursor = conn.cursor()

        # USAR LA MISMA LÓGICA QUE EL ORIGINAL
        load_dotenv()
        curso = os.getenv("QUERY_CURSOS_PATH")
        with open(curso, 'r', encoding='utf-8') as f:
            cursos = f.read()
        
        cursor.execute(cursos)  # Ejecutar la consulta unificada
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        print(f"Columnas en la consulta SQL: {columns}")

        if 'link_temario' in columns:
            link_id = columns.index('link_temario')
        else:
            link_id = None

        prioritized_columns = ['tecnologia_id', 'familia_id', 'nombre', 'clave']
            
        for i, row in enumerate(rows):
            row = tuple("NAA" if col is None else col for col in row)
            print(f"Row antes de construir lc_id: {row}")
            
            # Construir contenido
            content = ";".join(str(col) for col in row if col is not None)
            
            # Obtener datos de lc_id y asignar a la plantilla
            lc_id_value = [str(col) for col in row]  # Convertimos cada elemento de la fila a string
            lc_id_value = ";".join(lc_id_value)  # Unimos los elementos con punto y coma
            
            template = self.extract_data_from_lc_id(lc_id_value, namespace="Cursos")
                    
            # Convertir la plantilla a JSON y tokenizar
            template_str = json.dumps(template, ensure_ascii=False)
            tokens = self.milvus.tokenize(template_str, prioritized_columns=prioritized_columns)

            document = Document(
                page_content=content,  # contents
                metadata={
                    "context": content,
                    "tokens": tokens,
                    "orden": i,
                    "lc_id": lc_id_value,
                    # Agregar campos específicos para Milvus
                    "pais": template.get("pais", "NA"),
                    "pais_curso": template.get("pais_curso", "NA")
                }
            )
            setattr(document, "id", str(uuid.uuid4()))
            embeddings = self.milvus.openai_embeddings.embed_documents([content])
            
            # Usar MilvusVectorStore en lugar de Pinecone
            self.milvus.vector_store.add_documents(
                documents=[document], 
                embeddings=embeddings, 
                namespace="cursos"  # Todo va a la colección cursos ahora
            )
    
        conn.close()
    def clasificar_intencion_con_gpt(self, consulta: str) -> str:
        """Clasificar intención usando GPT - sin cambios"""
        prompt_clasificacion = f"""Clasifica el siguiente mensaje según su intención:
        
        Consulta: "{consulta}"
        
        Responde solo con "Cursos" si la consulta pregunta sobre un curso en general (por ejemplo, precio, disponibilidad, certificación, sesiones, etc.) ; en caso de que la consulta diga que curso esta relacionado a cierta certificacion o sólo se mencione el nombre de un posible curso (por ejemplo: java).
        Responde solo con "Certificaciones" si la consulta solicita certificaciones de algún tema o fabricante, no aplica si la consulta dice que curso esta asociado a cierta certificacion. 
        Responde solo con "Precio" si la consulta solicita precio o precios de cursos.
        Responde solo con "General" si la consulta es sobre un tema en general sin mencionar una clave especifica.
        Responde solo con "Agnostico" si la consulta hace referencia a la palabra agnostico o alguna de sus variantes o conjugaciones.
        Responde solo con "Temarios" si el mensaje pregunta específicamente por el temario, contenido, de que trata uno o mas cursos.
        Responde solo con "Recomendacion" si el mensaje solicita algún curso para cubrir algún tema o grupo de temas, por ejempo ("Cursos de administración de proyectos y agile")
        Responde solo con "Idioma" si la consulta pregunta específicamente por el idioma de un curso o cursos, incluyendo preguntas como "¿en qué idioma está?", "¿está en español?", "¿hay en inglés?", "idioma del curso", etc.
        Responde solo con "No" si la consulta no tiene relación con estos temas, si es demasiado general para determinar una intención específica o si no se cuenta con la información.
        Responde solo con "Laboratorio" si la consulta solicita información correspondiente a los laboratorios de los cursos (por ejemplo, precio, disponibilidad, país, clave, nombre)
        Responde solo con "Examenes" si la consulta solicita información correspondiente a los exámenes de los cursos (por ejemplo, precio, disponibilidad, país, clave, nombre)
        Intención:"""
        
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

        # Extrae la intención
        intencion = respuesta.content.strip()
        return intencion
        
    def format_language_response(self, data, query=""):
        """
        Función específica para responder consultas sobre idioma de cursos
        Respuesta concisa y directa sin plantilla completa
        """
        if not isinstance(data, dict):
            return "Error: No puedo determinar el idioma del curso"
        
        clave_curso = data.get('clave', 'NA')
        nombre_curso = data.get('nombre', 'NA')
        idioma_curso = data.get('idioma', 'Inglés')
        
        # Detectar si pregunta por curso específico o múltiples cursos
        query_lower = query.lower()
        
        # Si es una consulta específica sobre un curso
        if any(indicator in query_lower for indicator in ['el curso', 'este curso', clave_curso.lower()]):
            return f"El curso **{clave_curso}** ({nombre_curso}) está disponible en **{idioma_curso}**."
        
        # Si es una consulta más general
        return f"**{clave_curso}** - {nombre_curso}: **{idioma_curso}**"

    def rag(self, human_message: Union[str, HumanMessage], conversation_history=None):
        """
        Retrieval Augmented Generation prompt adaptado para Milvus.
        Toda la información de laboratorios y exámenes ya está en el namespace 'Cursos'
        CON INTEGRACIÓN DE FILTRADO POR PAÍS DEL USUARIO
        """
        # Función temporal para debugging
        def debug_curso_data(data, doc_metadata=None):
            """Función temporal para debugging de datos de curso"""
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
                patrones_sub = [' (Sub-Ext)', ' (Sub) (Digital)', ' (Sub)', '(Sub-Ext)', '(Sub) (Digital)', '(Sub)']
                
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
                Incluye el idioma del curso
                """
                if not isinstance(data, dict):
                    print('ERROR: data no es un diccionario')
                    return "Error: Datos de curso no válidos"
                    
                # DEBUGGING DETALLADO
                clave_curso = data.get('clave', '')
                subcontratado_field = data.get('subcontratado', 'No')
                idioma_curso = data.get('idioma', 'Inglés')  # Obtener idioma
                
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
                def safe_float_conversion(value, default="NA"):
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except Exception:
                        return default

                # Procesar información de laboratorios usando datos del curso principal
                clave_examen = data.get('clave_examen', 'NA')
                nombre_moneda_examenes=data.get('nombre_moneda_examenes','NA')
                clave_laboratorio = data.get('clave_laboratorio', 'NA')
                nombre_laboratorio = data.get('nombre_laboratorio', 'NA')
                costo_examen = data.get('costo_examen', 'NA')
                costo_laboratorio = data.get('costo_laboratorio', 'NA')
                re_take=data.get('incluye_re_take','NA')
                costo_re_take = safe_float_conversion(data.get('costo_re_take'))
                
                
                # Determinar si tiene laboratorio basado en los datos del curso
                if (clave_laboratorio and clave_laboratorio != 'NA' and clave_laboratorio != 'None' and
                    nombre_laboratorio and nombre_laboratorio != 'NA' and nombre_laboratorio != 'None'):
                    labs_info = f'{clave_laboratorio} / ${costo_laboratorio}{nombre_moneda_examenes}'
                else:
                    labs_info = "No lleva laboratorio"
                
                # Determinar si tiene examen de certificación basado en los datos del curso
                if clave_examen and clave_examen not in ['NA', 'None', '', 'NULL']:
                    examenes_info = f'{clave_examen} / ${costo_examen}{nombre_moneda_examenes}'
                else:
                    examenes_info = "No tiene certificación asociada"

                # Determinar disponibilidad
                if data.get('disponible', 'NA') == 'Si' and data.get('estatus_curso', '') == 'Liberado':
                    disponibilidad = "Habilitado"
                elif data.get('disponible', 'NA') == 'No' or data.get('estatus_curso', '') != 'Liberado':
                    disponibilidad = "En habilitación"
                else:
                    disponibilidad = "NA"

                #Determinar si incluye re take o no
                if re_take=='1' and costo_re_take !='NULL':
                    retake_info = f'Incluye re take con un costo de ${costo_re_take}'
                else:
                    retake_info = "No incluye re take"
                
                # PLANTILLA SIN BLOQUES DE CÓDIGO - FORMATO LIMPIO
                template = f"""**Curso**
            - **Clave**: {data.get('clave', 'NA')}
            - **Nombre**: {data.get('nombre', 'NA')} V{data.get('version', 'NA')}
            - **Tecnología/ Línea de negocio**: {data.get('tecnologia_id', 'NA')} / {data.get('linea_negocio_id', 'NA')}
            - **Entrega**: {data.get('entrega', 'NA')}
            - **Idioma**: {idioma_curso}
            - **Estatus del curso**: {disponibilidad}
            - **Tipo de curso**: {data.get('tipo_curso_id', 'NA')}
            - **Sesiones y Horas**: {data.get('sesiones', 'NA')} / {data.get('horas', 'NA')}
            - **Precio**: {data.get('precio', 'NA')} {data.get('nombre_moneda', 'NA')}
            - **Examen**: {examenes_info}
            - **Retake**:{retake_info}
            - **Laboratorio**: {labs_info}
            - **País**: {data.get('pais_curso', 'NA')}
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

            # === INTEGRACIÓN DE PAÍS DEL USUARIO - MEJORADA ===
            user_name = usuario_desde_query_params() or "desconocido"
            user_country = get_user_country(user_name)

            if user_country:
                print(f"País del usuario detectado: {user_country}")
                print(f"Aplicando filtro: cursos de {user_country} o LATAM con labs/exams en {user_country}")
            else:
                print("País del usuario NO detectado - mostrando todos los cursos")
            
            

            def ordenar_cursos(cursos):
                """Ordena los cursos con manejo robusto de errores"""
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

            # Funciones auxiliares
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
                words = query.lower().split()
                cleaned_words = []
                for word in words:
                    if word and word.strip() and word not in stop_words:
                        cleaned_words.append(word.strip())
                # Divide la consulta en palabras y elimina las stop words
                cleaned_queryes = " ".join(cleaned_words)
                if not cleaned_queryes.strip():
                    return query.strip()
                try:
                    cleaned_queryen = translate_with_mymemory(cleaned_queryes)
                except Exception as e:
                    print(f'Error en traducción:{e}')
                    cleaned_queryen = cleaned_queryes
                
                # devolver la consulta combinada
                if cleaned_queryes.strip() == cleaned_queryen.strip():
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
            - **Idioma**: [Español/Inglés - detectado automáticamente por la clave]
            - **Estatus del curso**: [valor]
            - **Tipo de curso**: [valor]
            - **Sesiones y Horas**: [valor] / [valor]
            - **Precio**: [valor] [moneda]
            - **Examen**: [valor]
            - **Retake**: [valor]
            - **Laboratorio**: [valor]
            - **País**: [valor]
            - **Complejidad**: [valor]
            - **Link al temario**: [valor]

            REGLAS CRÍTICAS:
            - SIEMPRE responde de forma amable y conversacional ANTES de mostrar la información del curso
            - USA EXACTAMENTE esta plantilla para cada curso que no sea subcontratado (SIN bloques de código ``` ```)
            - El campo **Idioma** se detecta automáticamente: si la clave contiene 'ESP' es Español, si no es Inglés
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
            - También informas sobre el idioma de los cursos basándote en su clave

            {TEMPLATE_OBLIGATORIA}
            """

                # INSTRUCCIONES ESPECÍFICAS POR INTENCIÓN
                specific_instructions = {
                    "Idioma": """
            **Instrucciones para consultas sobre IDIOMA**
            - SIEMPRE responde de forma amable : ¡Por supuesto! Te comparto la información del idioma:
            - Para consultas específicas sobre un curso: "El curso [CLAVE] ([NOMBRE]) está disponible en [IDIOMA]."
            - Para múltiples cursos: mostrar lista concisa: "[CLAVE] - [NOMBRE]: [IDIOMA]"
            - NO uses la plantilla completa del curso, sólo responde la información del idioma solicitada
            - Si preguntan por cursos en español específicamente: filta y muestra sólo los que contengan 'ESP' en la clave
            - Si preguntan por cursos en inglés: muestra los que NO contengan 'ESP' en la clave
            - Mantén un tono conversacional y amable
            
            """,
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

            namespace_map = {
                "Temarios": "temarios"
            }
            namespace = namespace_map.get(intencion, "cursos")
            leader = get_unified_leader_prompt(intencion)    

            enhanced_query = f"{clean_query(human_message.content)}"  
            if not enhanced_query.strip():
                print("Consulta vacía, se omite la búsqueda en Milvus.")
                return "Disculpa, no puedo procesar una consulta vacía. Por favor, intenta nuevamente."

            # Realizar la búsqueda SOLO en el namespace principal usando Milvus CON FILTROS DE PAÍS CONDICIONALES
            try:
                # NUEVO: Aplicar filtros de país solo si es apropiado
                if user_country and namespace == "cursos":
                    documents = self.retriever(
                        namespace=namespace, 
                        query=enhanced_query,
                        pais_usuario=user_country,  # parámetro simplificado
                    ).get_relevant_documents(query=enhanced_query)
                else:
                    # Sin filtros de país para evitar 0 resultados
                    documents = self.retriever(
                        namespace=namespace, 
                        query=enhanced_query
                    ).get_relevant_documents(query=enhanced_query)
                
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
                            lc_id = "No metadata"
                            subcontratado_meta = "No metadata"
                            
                        # Extraer datos para ver la clave
                        try:
                            data = self.extract_data_from_lc_id(lc_id, namespace="Cursos")
                            if isinstance(data, dict):
                                clave = data.get('clave', 'Sin clave')
                                subcontratado_data = data.get('subcontratado', 'Sin dato')
                                
                                print(f"\nDOC {i+1}:")
                                print(f"  Clave: {clave}")
                                print(f"  Subcontratado (metadata): {subcontratado_meta}")
                                print(f"  Subcontratado (data): {subcontratado_data}")
                                print(f"  Tiene (Sub) en clave: {'(sub)' in clave.lower() or '(SUB)' in clave}")
                                
                                tiene_sub_patron = any(patron in clave.upper() for patron in [' (SUB)', ' (SUB-EXT)', ' (Sub) (Digital)', ' (Sub-Ext)', ' (sub)', ' (Sub)'])
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
                print(f"Error al realizar la búsqueda en Milvus: {e}")
                logging.error(f"Error al realizar la búsqueda en Milvus: {e}")
                return "Disculpa, hubo un problema al realizar la búsqueda. Intenta nuevamente."
            
            # Para búsquedas de temarios, mantenemos la lógica existente pero adaptada a Milvus
            if intencion == "Recomendacion" or intencion == "Cursos" or intencion == "Agnostico": 
                additional_namespace = "temarios" if namespace == "cursos" else "cursos"
                
                # CORREGIDO: Aplicar filtros condicionalmente
                if user_country and additional_namespace == "cursos":
                    additional_documents = self.retriever(
                        namespace=additional_namespace,
                        query=enhanced_query,
                        pais_usuario=user_country
                    ).get_relevant_documents(query=enhanced_query)
                else:
                    additional_documents = self.retriever(
                        namespace=additional_namespace,
                        query=enhanced_query
                    ).get_relevant_documents(query=enhanced_query)
                
                # Lógica de extracción de claves permanece igual...
                def extraer_clave(document):
                    match = re.search(r"Clave:\s*\*\*(.*?)\*\*", document.metadata.get('lc_id', ''))
                    return match.group(1) if match else None

                claves = [extraer_clave(doc) for doc in additional_documents]
                claves = [clave for clave in claves if clave is not None]
                base_claves = [clave.split()[0] for clave in claves]
                
                contador_claves = Counter(base_claves)
                claves_ordenadas = [clave for clave, _ in contador_claves.most_common()]
                claves_string = ', '.join(claves_ordenadas)
                
                if claves_string.strip():
                    # CORREGIDO: Aplicar filtros condicionalmente
                    if user_country:
                        additional_doc_temarios = self.retriever(
                            namespace="cursos",
                            query=claves_string,
                            pais_usuario=user_country
                        ).get_relevant_documents(query=claves_string)
                    else:
                        additional_doc_temarios = self.retriever(
                            namespace="cursos",
                            query=claves_string
                        ).get_relevant_documents(query=claves_string)
                    documents = additional_doc_temarios + documents
                
                documents_cursos = documents
                documents.extend(additional_documents)  

            if not documents:
                # FALLBACK: Si no hay documentos con filtros, intentar sin filtros
                if user_country:
                    print(f"No se encontraron documentos para {user_country}. Reintentando sin filtros...")
                    documents = self.retriever(
                        namespace=namespace, 
                        query=enhanced_query
                    ).get_relevant_documents(query=enhanced_query)
                
                # Si aún no hay documentos, respuesta por defecto
                if not documents:
                    return "Disculpa, no tengo la información que pides. Te sugiero contactar a un Ing. Preventa para más detalles." 

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
                        debug_curso_data(data, doc.metadata)
                        
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
            # Logging permanece igual pero ahora incluye información del usuario
            def usuario():
                global user_name
                try:
                    # Corregido: acceso correcto a query_params en Streamlit
                    if hasattr(st, 'query_params'):
                        user_name_param = st.query_params.get('user_name', None)
                        if user_name_param:
                            return user_name_param if isinstance(user_name_param, str) else str(user_name_param)
                    return st.session_state.get('user_name', None)
                except Exception as e:
                    print(f"Error obteniendo user_name: {e}")
                    return None

            user_name = usuario() or "desconocido"

            log_entry = {
                "user_name": user_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "question": human_message.content,
                "response": response_content
            }

            # CORREGIDO: Sin referencia a paises_filtro
            if user_country:
                print(f"Log info - Usuario: {user_name}, País: {user_country}, Filtro aplicado: pais_usuario='{user_country}'")
            else:
                print(f"Log info - Usuario: {user_name}, País: No detectado, Sin filtros")

            log_interaction(**log_entry)
            
        except Exception as e:
            print(f"ERROR CAPTURADO EN RAG: {e}")
            print(f"Tipo: {type(e)}")
            import traceback
            traceback.print_exc()
            return f"Error durante la transmisión de respuesta: {str(e)}"

# Funciones auxiliares para migración y testing (sin cambios)
def migrate_from_pinecone_to_milvus():
    """
    Función auxiliar para migrar datos de Pinecone a Milvus
    (si necesitas migrar datos existentes)
    """
    print("📄 Función de migración disponible")
    print("Esta función te ayudaría a migrar datos existentes de Pinecone a Milvus")
    print("Ya tienes tus datos cargados en Milvus, así que no es necesario ejecutarla")

#==============================================================================================
#FUNCIONES DE TESTING
#==============================================================================================
# def test_milvus_connection():
#     """Función de prueba para verificar la conexión a Milvus"""
#     try:
#         retriever = HybridSearchRetriever()
#         print("Conexión a Milvus exitosa")
        
#         # Verificar colecciones
#         collections = utility.list_collections()
#         print(f"Colecciones disponibles: {collections}")
        
#         for collection_name in ["cursos", "temarios"]:
#             if collection_name in collections:
#                 collection = Collection(collection_name)
#                 collection.load()
#                 count = collection.num_entities
#                 print(f"{collection_name}: {count} registros")
#             else:
#                 print(f"{collection_name}: no encontrada")
        
#         return True
        
#     except Exception as e:
#         print(f"Error en conexión: {e}")
#         return False

# def test_simple_search(query="¿Tenemos el curso CSA?"):
#     """Función de prueba para realizar una búsqueda simple"""
#     try:
#         retriever = HybridSearchRetriever()
        
#         # Realizar búsqueda
#         documents = retriever.retriever(namespace="cursos", query=query).get_relevant_documents(query=query)
        
#         print(f"Búsqueda: '{query}'")
#         print(f"Documentos encontrados: {len(documents)}")
        
#         if documents:
#             first_doc = documents[0]
#             content_preview = first_doc.page_content[:100] + "..." if len(first_doc.page_content) > 100 else first_doc.page_content
#             print(f"Primer resultado: {content_preview}")
#             print(f"Score: {first_doc.metadata.get('score', 'N/A')}")
        
#         return True
        
#     except Exception as e:
#         print(f"Error en búsqueda: {e}")
#         return False

# def test_country_filtering():
#     """NUEVA: Función de prueba para verificar el filtrado por país - CORREGIDA"""
#     try:
#         retriever = HybridSearchRetriever()
        
#         # Probar con filtro de país - SINTAXIS CORREGIDA
#         query = "¿Tenemos el curso CSA?"
#         paises_test = ["LATAM", "MX"]
        
#         print(f"Probando filtro con países: {paises_test}")
        
#         # Probar primero sin filtros
#         documents_sin_filtro = retriever.retriever(
#             namespace="cursos", 
#             query=query
#         ).get_relevant_documents(query=query)
        
#         print(f"Documentos SIN filtro: {len(documents_sin_filtro)}")
        
#         # Ahora probar con filtros
#         documents_con_filtro = retriever.retriever(
#             namespace="cursos", 
#             query=query,
#             paises=paises_test
#         ).get_relevant_documents(query=query)
        
#         print(f"Documentos CON filtro: {len(documents_con_filtro)}")
        
#         # Verificar que los documentos tienen los países correctos
#         if documents_con_filtro:
#             for i, doc in enumerate(documents_con_filtro[:3]):  # Mostrar solo los primeros 3
#                 pais = doc.metadata.get('pais', 'N/A')
#                 pais_curso = doc.metadata.get('pais_curso', 'N/A')
#                 print(f"  Doc {i+1}: país={pais}, país_curso={pais_curso}")
#         else:
#             print("No se encontraron documentos con filtro")
        
#         # Considerar exitoso si hay documentos sin filtro (problema de sintaxis resuelto)
#         return len(documents_sin_filtro) > 0
        
#     except Exception as e:
#         print(f"Error en prueba de filtrado por país: {e}")
#         return False

# Función principal para testing completo
# def run_complete_migration_test():
#     """Ejecutar prueba completa de migración"""
#     print("Iniciando prueba completa de migración a Milvus con filtrado por país")
#     print("=" * 60)
    
#     tests = [
#         ("Crear colecciones si no existen", create_milvus_collections_if_not_exist),
#         ("Verificar conexión a Milvus", test_milvus_connection),
#         ("Realizar búsqueda simple", test_simple_search),
#         ("Probar filtrado por país", test_country_filtering)  # NUEVA prueba
#     ]
    
#     all_passed = True
    
#     for test_name, test_func in tests:
#         print(f"\nEjecutando: {test_name}")
#         try:
#             result = test_func()
#             if result:
#                 print(f"{test_name}: EXITOSO")
#             else:
#                 print(f"{test_name}: FALLÓ")
#                 all_passed = False
#         except Exception as e:
#             print(f"{test_name}: ERROR - {e}")
#             all_passed = False
    
#     print("\n" + "=" * 60)
#     if all_passed:
#         print("¡Migración a Milvus con filtrado por país completada exitosamente!")
#         print("Tu sistema está listo para usar con Milvus y filtrado por país del usuario")
#     else:
#         print("Algunos tests fallaron. Revisa los errores antes de continuar.")
    
#     return all_passed

def create_milvus_collections_if_not_exist():
    """Crear colecciones de Milvus si no existen"""
    try:
        from pymilvus import FieldSchema, CollectionSchema, DataType
        
        # Conectar a Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        
        collections_to_create = ["cursos", "temarios"]
        
        for collection_name in collections_to_create:
            if collection_name not in utility.list_collections():
                print(f"Creando colección '{collection_name}'...")
                
                # Definir esquema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # text-embedding-3-small
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2500),
                    FieldSchema(name="pais", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="pais_curso", dtype=DataType.VARCHAR, max_length=64)
                ]
                
                schema = CollectionSchema(fields, f"Colección {collection_name} para cursos NETEC")
                collection = Collection(collection_name, schema)
                
                # Crear índice vectorial
                index_params = {
                    "metric_type": "IP",  # Inner Product
                    "index_type": "IVF_FLAT", 
                    "params": {"nlist": 512}
                }
                collection.create_index("embedding", index_params)
                
                print(f"Colección '{collection_name}' creada exitosamente")
            else:
                print(f"Colección '{collection_name}' ya existe")
        
        return True
        
    except Exception as e:
        print(f"Error creando colecciones: {e}")
        return False


# Configuraciones adicionales para compatibilidad
class MilvusConfig:
    """Configuración centralizada para Milvus"""
    
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_names = {
            "cursos": "cursos",
            "temarios": "temarios"
        }
        self.embedding_dim = 1536  # text-embedding-3-small
        self.max_text_length = 2500
        self.search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        self.index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 512}
        }

# Instancia global de configuración
milvus_config = MilvusConfig()
def test_country_filtering_complete():
    """
    Función completa para probar el filtrado por país
    Simula diferentes países y verifica resultados
    """
    print("="*80)
    print("INICIANDO PRUEBAS DE FILTRADO POR PAÍS")
    print("="*80)
    
    retriever = HybridSearchRetriever()
    
    # Casos de prueba
    test_cases = [
        {
            "pais": "México",
            "nombre": "México",
            "query": "¿Tenemos el curso CSA?",
            "esperado": "Solo cursos MX o LATAM con labs/exams en MX"
        },
        {
            "pais": "Colombia",
            "nombre": "Colombia", 
            "query": "¿Tenemos el curso CSA?",
            "esperado": "Solo cursos CO o LATAM con labs/exams en CO"
        },
        {
            "pais": "Perú",
            "nombre": "Perú",
            "query": "¿Tenemos el curso CSA?",
            "esperado": "Solo cursos PE o LATAM con labs/exams en PE"
        },
        {
            "pais": None,
            "nombre": "Sin país",
            "query": "¿Tenemos el curso CSA?",
            "esperado": "Todos los cursos sin filtro"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"PRUEBA {i}: Usuario de {test['nombre']}")
        print(f"{'='*80}")
        print(f"Query: {test['query']}")
        print(f"País simulado: {test['pais']}")
        print(f"Resultado esperado: {test['esperado']}")
        print(f"{'-'*80}")
        
        try:
            # Realizar búsqueda con país simulado
            documents = retriever.retriever(
                namespace="cursos",
                query=test['query'],
                pais_usuario=test['pais']
            ).get_relevant_documents(query=test['query'])
            
            print(f"Documentos encontrados: {len(documents)}")
            
            if len(documents) > 0:
                print(f"\nPrimeros 10 resultados:")
                print(f"{'#':<4} {'Clave':<20} {'País':<10} {'País Curso':<15}")
                print(f"{'-'*60}")
                
                # Contadores para validación
                paises_encontrados = set()
                paises_curso_encontrados = set()
                
                for idx, doc in enumerate(documents[:10], 1):
                    try:
                        # Extraer datos del documento
                        lc_id = doc.metadata.get('lc_id', '')
                        data = retriever.extract_data_from_lc_id(lc_id, namespace="Cursos")
                        
                        clave = data.get('clave', 'N/A')[:18]
                        pais = data.get('pais', 'N/A')
                        pais_curso = data.get('pais_curso', 'N/A')
                        
                        paises_encontrados.add(pais)
                        paises_curso_encontrados.add(pais_curso)
                        
                        print(f"{idx:<4} {clave:<20} {pais:<10} {pais_curso:<15}")
                        
                    except Exception as e:
                        print(f"{idx:<4} Error procesando documento: {e}")
                
                # Validación
                print(f"\nRESUMEN:")
                print(f"   Países únicos encontrados: {sorted(paises_encontrados)}")
                print(f"   Países de curso únicos: {sorted(paises_curso_encontrados)}")
                
                # Verificar si el filtro funcionó correctamente
                if test['pais']:
                    pais_esperado = test['pais']
                    
                    # Verificar lógica: pais_curso == pais_esperado OR (pais_curso == LATAM AND pais == pais_esperado)
                    cursos_invalidos = []
                    for doc in documents[:10]:
                        try:
                            lc_id = doc.metadata.get('lc_id', '')
                            data = retriever.extract_data_from_lc_id(lc_id, namespace="Cursos")
                            pais = data.get('pais', 'N/A')
                            pais_curso = data.get('pais_curso', 'N/A')
                            
                            # Validar regla de negocio
                            es_valido = (
                                pais_curso == pais_esperado or 
                                (pais_curso == "LATAM" and pais == pais_esperado)
                            )
                            
                            if not es_valido:
                                cursos_invalidos.append({
                                    'clave': data.get('clave', 'N/A'),
                                    'pais': pais,
                                    'pais_curso': pais_curso
                                })
                        except:
                            pass
                    
                    if cursos_invalidos:
                        print(f"\nFALLO - Cursos que NO deberían aparecer:")
                        for curso in cursos_invalidos:
                            print(f"   - {curso['clave']}: pais={curso['pais']}, pais_curso={curso['pais_curso']}")
                    else:
                        print(f"\nÉXITO - Todos los cursos cumplen el filtro correctamente")
                else:
                    print(f"\nSin filtro - Mostrando cursos de todos los países (esperado)")
                    
            else:
                print(f"\nADVERTENCIA - No se encontraron documentos")
                
        except Exception as e:
            print(f"\nERROR en prueba: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("PRUEBAS COMPLETADAS")
    print(f"{'='*80}\n")


def test_country_quick(pais_codigo, query="¿Tenemos el curso CSA?"):
    """
    Prueba rápida para un país específico
    
    Args:
        pais_codigo: Código del país (ej: "MX", "CO", "PE")
        query: Query de búsqueda
    """
    print(f"\nPrueba rápida - País: {pais_codigo} | Query: '{query}'")
    print("-"*60)
    
    retriever = HybridSearchRetriever()
    
    documents = retriever.retriever(
        namespace="cursos",
        query=query,
        pais_usuario=pais_codigo
    ).get_relevant_documents(query=query)
    
    print(f"Resultados: {len(documents)} cursos\n")
    
    if documents:
        print(f"{'#':<4} {'Clave':<25} {'País':<10} {'País Curso':<12}")
        print("-"*60)
        
        for i, doc in enumerate(documents[:10], 1):
            try:
                lc_id = doc.metadata.get('lc_id', '')
                data = retriever.extract_data_from_lc_id(lc_id, namespace="Cursos")
                
                clave = data.get('clave', 'N/A')[:23]
                pais = data.get('pais', 'N/A')
                pais_curso = data.get('pais_curso', 'N/A')
                
                print(f"{i:<4} {clave:<25} {pais:<10} {pais_curso:<12}")
                
            except Exception as e:
                print(f"{i:<4} Error: {e}")


def test_user_from_database(first_name, last_name, query="cursos de python"):
    """
    Prueba con un usuario real de la base de datos
    
    Args:
        first_name: Nombre del usuario
        last_name: Apellido del usuario
        query: Query de búsqueda (opcional)
    """
    user_name = f"{first_name} {last_name}"
    print("\n" + "="*80)
    print(f" PRUEBA CON USUARIO REAL: {user_name}")
    print("="*80)
    
    # Obtener país del usuario desde SQL
    print(f"\n Buscando usuario en base de datos...")
    print(f"   Nombre: {first_name}")
    print(f"   Apellido: {last_name}")
    print(f"   Email: buscando en @netec...")
    
    country = get_user_country(user_name)
    
    if country:
        print(f" País detectado: {country}")
        print(f" Query de prueba: '{query}'")
        print("-"*80)
        
        # Ejecutar búsqueda con el país del usuario
        retriever = HybridSearchRetriever()
        
        documents = retriever.retriever(
            namespace="cursos",
            query=query,
            pais_usuario=country
        ).get_relevant_documents(query=query)
        
        print(f"\n Resultados: {len(documents)} cursos encontrados\n")
        
        if documents:
            print(f"{'#':<4} {'Clave':<25} {'País':<15} {'País Curso':<12}")
            print("-"*65)
            
            for i, doc in enumerate(documents[:15], 1):
                try:
                    lc_id = doc.metadata.get('lc_id', '')
                    data = retriever.extract_data_from_lc_id(lc_id, namespace="Cursos")
                    
                    clave = data.get('clave', 'N/A')[:23]
                    pais = data.get('pais', 'N/A')
                    pais_curso = data.get('pais_curso', 'N/A')
                    
                    # Marcar si el curso cumple con el filtro
                    es_valido = (pais_curso == country) or (pais_curso == "LATAM" and pais == country)
                    marca = "ok" if es_valido else "nok"
                    
                    print(f"{i:<4} {clave:<25} {pais:<15} {pais_curso:<12} {marca}")
                    
                except Exception as e:
                    print(f"{i:<4} Error: {e}")
            
            if len(documents) > 15:
                print(f"\n... y {len(documents) - 15} cursos más")
            
            # Validación
            print(f"\n VALIDACIÓN:")
            cursos_invalidos = 0
            for doc in documents:
                try:
                    lc_id = doc.metadata.get('lc_id', '')
                    data = retriever.extract_data_from_lc_id(lc_id, namespace="Cursos")
                    pais = data.get('pais', 'N/A')
                    pais_curso = data.get('pais_curso', 'N/A')
                    
                    es_valido = (pais_curso == country) or (pais_curso == "LATAM" and pais == country)
                    if not es_valido:
                        cursos_invalidos += 1
                except:
                    pass
            
            if cursos_invalidos == 0:
                print(f"    ÉXITO - Todos los cursos pertenecen a {country}")
            else:
                print(f"    FALLO - {cursos_invalidos} cursos no deberían aparecer")
        else:
            print("   No se encontraron cursos para este usuario")
            
    else:
        print(f" No se pudo detectar el país del usuario '{user_name}'")
        print("   Posibles causas:")
        print("   - El usuario no existe en la base de datos")
        print("   - El email no contiene '@netec'")
        print("   - No tiene país asignado (pais_id es NULL)")


if __name__ == "__main__":
    """
    Bloque principal - Solo se ejecuta cuando se corre el archivo directamente
    En producción (importado por Streamlit), este bloque NO se ejecuta
    """
    print("🔧 Sistema Hybrid Search Retriever con Milvus")
    print("=" * 60)
    print("✅ Módulo cargado correctamente")
    print("📝 Para ejecutar pruebas, usa:")
    print("   python -m models.hybrid_milvus test")
    print("   python -m models.hybrid_milvus user NombreUsuario Apellido")
    print("   python -m models.hybrid_milvus México")

   