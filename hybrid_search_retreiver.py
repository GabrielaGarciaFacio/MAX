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
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from pymilvus import connections, Collection
from langchain_openai import OpenAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from .conf import settings
import tempfile
import requests
import re
from pydantic import BaseModel
import os
import os, datetime

# Modo debug: imprime documentos completos recuperados
DEBUG_RETRIEVAL = True

def dump_docs(namespace: str, docs: list, max_docs: int | None = None):
    if not DEBUG_RETRIEVAL:
        return
    total = len(docs)
    n = total if max_docs is None else min(max_docs, total)
    print(f"\n===== {namespace}: imprimiendo {n}/{total} documentos =====")
    for i, d in enumerate(docs[:n], 1):
        score = d.metadata.get("score")
        col   = d.metadata.get("collection")
        preview = (d.page_content or "")[:300].replace("\n", " ")
        print("\n" + "=" * 80)
        print(f"[{namespace} #{i}] score={score} | collection={col} | len(text)={len(d.page_content or '')}")
        print("-" * 80)
        print(preview + ("..." if len(d.page_content or "") > 300 else ""))
        print("=" * 80)


def save_docs(namespace: str, docs: list, out_dir: str = "./debug_retrieval", max_docs: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    n = len(docs) if max_docs is None else min(max_docs, len(docs))
    for i, d in enumerate(docs[:n], 1):
        fname = os.path.join(out_dir, f"{ts}_{namespace}_{i:02d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(d.page_content if isinstance(d.page_content, str) else str(d.page_content))
    print(f"[DEBUG] Guardados {n} documentos en {out_dir} ({namespace})")

def debug_save_prompt(tag: str, content, out_dir: str = "./debug_prompts"):
    """Imprime y guarda el prompt en un archivo con timestamp, convirtiendo a str lo que sea."""
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"[DEBUG] No se pudo crear {out_dir}: {e}")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_text(obj) -> str:
        if obj is None:
            return ""
        # Mensajes de LangChain (SystemMessage/HumanMessage/etc.)
        if hasattr(obj, "content"):
            try:
                return str(obj.content)
            except Exception:
                return str(obj)
        # Texto ya es texto
        if isinstance(obj, str):
            return obj
        # Estructuras comunes
        if isinstance(obj, dict):
            try:
                return json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                return str(obj)
        if isinstance(obj, (list, tuple, set)):
            # Si es lista de strings, haz join; si no, a JSON
            try:
                if all(isinstance(x, str) for x in obj):
                    return "\n".join(obj)
                return json.dumps(list(obj), ensure_ascii=False, indent=2)
            except Exception:
                return "\n".join(map(str, obj))
        # Cualquier otra cosa
        return str(obj)

    text = to_text(content)
    path = os.path.join(out_dir, f"{ts}_{tag}.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[DEBUG] Prompt '{tag}' guardado en: {os.path.abspath(path)}")
    except Exception as e:
        print(f"[DEBUG] No se pudo guardar prompt '{tag}': {e}")

    # También lo imprimimos completo en consola
    print("\n" + "="*80)
    print(f"===== {tag} =====")
    print(text)
    print(f"===== FIN {tag} =====")
    print("="*80 + "\n")

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MilvusRetriever:
    def __init__(self, host: str, port: int, collection_name: str, top_k: int = 40):
        self.top_k = top_k
        self.collection_name = collection_name
        connections.connect(alias="default", host=host, port=port)
        self.col = Collection(collection_name)
        self.col.load()
        self.search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        self.emb = OpenAIEmbeddings(model="text-embedding-3-small")

        try:
            self.available_fields = {f.name for f in self.col.schema.fields}
            print(f"[Milvus:{self.collection_name}] Campos:", self.available_fields)
        except Exception as e:
            self.available_fields = set()
            print(f"[Milvus:{self.collection_name}] No se pudo leer el esquema: {e}")

    def _hit_to_text(self, hit):
        # intenta varias formas según la versión de pymilvus
        txt = ""
        try:
            ent = getattr(hit, "entity", None)
            if ent and hasattr(ent, "get"):
                txt = ent.get("text") or ""
            if not txt and hasattr(hit, "fields"):
                fields = getattr(hit, "fields") or {}
                if isinstance(fields, dict):
                    txt = fields.get("text") or txt
            if not txt and hasattr(hit, "to_dict"):
                d = hit.to_dict()
                txt = (d.get("entity", {}) or {}).get("text", "") or d.get("text", "") or txt
        except Exception:
            pass
        return txt or ""

    def get_relevant_documents(self, query: str, metadata_filter: dict | None = None):
        try:
            vec = self.emb.embed_query(query)
            output_fields = ["text"] if "text" in self.available_fields else []
            res = self.col.search(
                data=[vec],
                anns_field="embedding",
                param=self.search_params,
                limit=self.top_k,
                output_fields=output_fields,
            )
        except Exception as e:
            print(f"[Milvus] search error in '{self.collection_name}': {e}")
            return []

        if not res or not res[0]:
            return []

        docs = []
        for i, hit in enumerate(res[0], 1):
            txt = self._hit_to_text(hit)
            if not txt:
                print(f"[WARN] {self.collection_name} Hit#{i} sin 'text' (len=0). Revisa datos de carga.")
            docs.append(Document(
                page_content=txt,
                metadata={
                    "score": float(getattr(hit, "distance", 0.0)),
                    "collection": self.collection_name
                }
            ))

        # Filtrado por país (buscando la línea "País: X" en el texto)
        if metadata_filter and "pais" in metadata_filter:
            val = metadata_filter["pais"]
            if isinstance(val, dict):
                val = val.get("$eq") or next(iter(val.values()), None)
            if val:
                import re
                pattern = re.compile(rf"(?mi)^\s*Pa[ií]s\s*:\s*{re.escape(str(val))}\s*$")
                docs = [d for d in docs if d.page_content and pattern.search(d.page_content)]
        return docs


os.environ['PYTHONUTF8'] = '1'
class CustomPromptModel(BaseModel):
    prompt_template: PromptTemplate

    class Config:
        arbitrary_types_allowed = True

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_PREFIX = ""  # el mismo prefijo que usas al cargar

COLLECTION_MAP = {
    "Cursos":       f"{COLLECTION_PREFIX}cursos",
    "Laboratorios": f"{COLLECTION_PREFIX}laboratorios",
    "Examenes":     f"{COLLECTION_PREFIX}examenes",
    "Temarios":     f"{COLLECTION_PREFIX}temarios",
}

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

class HybridSearchRetriever:
    _chat: ChatOpenAI = None

    def _index_temarios(self, tem_docs):
        idx = {}
        for d in tem_docs:
            t = self.extract_data_from_text(d.page_content, "Temarios")
            base = (t.get("clave") or "").split()[0]
            if base and t.get("link_temario"):
                idx.setdefault(base, t)  # guarda el primero que aparezca
        return idx

    def __init__(self):
        set_llm_cache(InMemoryCache())
        self.message_history = []

    def add_to_history(self, message: BaseMessage):
        self.message_history.append(message)

    def format_history(self):
        max_interactions = 5
        return " ".join([
            f"Usuario: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
            for msg in self.message_history[-max_interactions:]
        ])

    def get_history(self):
        return self.message_history

    def retriever(self, namespace: str = "Cursos", top_k: int = 40) -> MilvusRetriever:
        collection_name = COLLECTION_MAP.get(namespace, COLLECTION_MAP["Cursos"])
        return MilvusRetriever(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=collection_name,
            top_k=top_k,
        )


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
 
    def pdf_loader(self):
        '''
        Embed PDF from SQL database
        1. Connect to SQL database to retrieve PDF links
        2. Extract text from each PDF url
        3. Split into pages
        4. Embed each page 
        5. Store in Pinecone (upsert)
        '''
    #self.initialize()
        connectionString =("DRIVER={ODBC Driver 18 for SQL Server};""SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;" "DATABASE=netec_prod;""UID=netec_read;""PWD=R3ad55**N3teC+*;""TrustServerCertificate=yes;")
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

    def extract_data_from_text(self, text: str, namespace: str = "Cursos") -> dict:
        """
        Parsea 'page_content' generado por record_to_text() / temarios en el loader.
        Devuelve un dict con llaves similares a las que usabas antes.
        """
        text = text or ""
        # Mapa de etiquetas -> llave destino (cursos/labs/examenes)
        if namespace == "Temarios":
            # Primera línea: "Clave: ... | Nombre: ... | Link: ..."
            first = text.splitlines()[0] if text else ""
            m_clave  = re.search(r"Clave:\s*(.*?)\s*(?:\||$)", first)
            m_nombre = re.search(r"Nombre:\s*(.*?)\s*(?:\||$)", first)
            m_link   = re.search(r"Link:\s*(.*?)\s*(?:$)", first)
            return {
                "clave": (m_clave.group(1).strip() if m_clave else "NA"),
                "nombre": (m_nombre.group(1).strip() if m_nombre else "NA"),
                "link_temario": (m_link.group(1).strip() if m_link else "Temario no encontrado"),
                "temario_texto": "\n".join(text.splitlines()[2:]).strip() if text else ""
            }

        # etiquetas comunes
        pairs = [
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
        ] if namespace == "Cursos" else [
            ("Clave", "clave"), ("Nombre", "nombre"),
            ("Clave examen", "clave_Examen"), ("Tipo elemento", "tipo_elemento"),
            ("Nombre examen", "nombre_examen"), ("Base costo", "base_costo"),
            ("País", "pais"), ("Costo", "costo"),
        ]

        out = {dst: "NA" for _, dst in pairs}
        for label, dst in pairs:
            # captura hasta fin de línea
            m = re.search(rf"(?mi)^\s*{re.escape(label)}\s*:\s*(.*?)\s*$", text)
            if m:
                out[dst] = m.group(1).strip()

        # Normaliza link_temario si viene “Temario no encontrado”
        if out.get("link_temario") in (None, "", "NA"):
            out["link_temario"] = "Temario no encontrado"

        return out


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
    def load_sql(self, namespace):#self,sql,namespace="Labs"
        #self.initialize()
       
        #Connect to the bd
        connectionString =("DRIVER={ODBC Driver 18 for SQL Server};""SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;" "DATABASE=netec_prod;""UID=netec_read;""PWD=R3ad55**N3teC+*;""TrustServerCertificate=yes;")
        conn=pyodbc.connect(connectionString)
        cursor=conn.cursor()
 
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
        if namespace=="Cursos":
            sql=cursos
        elif namespace=="Laboratorios":
            sql=laboratorios
        elif namespace=="Examenes":
            sql=examenes
        else:
            sql=cursos

        cursor.execute(sql) #cursos laboratorios examenes
        rows=cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        print(f"Columnas en la consulta SQL: {columns}")

        # si no aparece, fallback a None
        pais_idx = columns.index('pais') if 'pais' in columns else None

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

            # extrae el pais directamente del row:
            if pais_idx is not None:
                pais_value = row[pais_idx]
            else:
                pais_value = "NA"

            # Obtener datos de lc_id y asignar a la plantilla
            #lc_id_value = row[columns.index('lc_id')] if 'lc_id' in columns else ''
            lc_id_value = [str(col) for col in row]  # Convertimos cada elemento de la fila a string
            lc_id_value = ";".join(lc_id_value)  # Unimos los elementos con punto y coma
            #print(f"LC_ID value from row: {lc_id_value}") 
            template = self.extract_data_from_text(document.page_content, namespace="Cursos")
                       
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
                    "lc_id": lc_id_value,
                    "pais": pais_value or "NA"
                    
            })
            embeddings=self.pinecone.openai_embeddings.embed_documents([content])
            self.pinecone.vector_store.add_documents(documents=[document],embeddings=embeddings, namespace=namespace)#Cursos Examenes Laboratorios
       
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
        def usuario():
            global user_name
            params = st.query_params
            user_name = "Gustavo Olarte"#Natalia Gómez #params.get('user_name', [None])[:]       

        usuario()

        def get_user_country(user_name: str) -> str | None:
            # 1) Partir el full name en first_name y last_name
            names = user_name.strip().split(" ", 1)
            first_name = names[0]
            last_name  = names[1] if len(names) > 1 else ""

            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                "SERVER=scenetecprod.czbotsckvb07.us-west-2.rds.amazonaws.com;"
                "DATABASE=netec_prod;"
                "UID=netec_read;"
                "PWD=R3ad25**SC3.2025-;"
                "TrustServerCertificate=yes;"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            sql = """
            SELECT p.nombre AS pais
            FROM [netec_prod].[dbo].[users] u
            INNER JOIN [netec_prod].[dbo].[paises] p
                ON u.pais_id = p.id
            WHERE
                u.first_name = ?
            AND u.last_name  = ?
            AND u.email LIKE '%netec%';
            """
            cursor.execute(sql, first_name, last_name)
            row = cursor.fetchone()
            conn.close()
            return row.pais if row else None
        
        # Averigua el país del usuario
        user_country = get_user_country(user_name)

        def format_response(data, labs_data=None, examenes_data=None): 
            
            if isinstance(labs_data, tuple):
                labs_data = { "nombre_examen": labs_data[0], "clave_examen": labs_data[1], "costo": labs_data[2] }

            if isinstance(examenes_data, tuple):
                examenes_data = { "nombre_examen": examenes_data[0], "clave_examen": examenes_data[1], "costo": examenes_data[2] }
            
            if "(sub)" in data.get('clave', '').lower():
                return f"""
                **Curso Subcontratado**
                - Clave:{data.get('clave','NA')}
                - Nombre: {data.get('nombre','NA')}
                - Más información: link :)
                """
            
            #2. formateamos link_temario y version
            
            link_temario = data.get('link_temario', 'Temario no encontrado')
            if link_temario and link_temario != 'Temario no encontrado':
                if link_temario.startswith("https://"):
                    s_link_temario = link_temario.replace(" ", "%20")
                    link_temario = f"[Link al temario]({s_link_temario})"
                else:
                    link_temario = 'Temario no encontrado'
            
            
            version = data.get('version', 'NA')
            version_number = version.split()[0] if version else 'NA'
            enfoque = 'Teórico' if data.get('tipo_elemento', 'NA').startswith('Labo') else 'Práctico'


            
           #3. Formatear la información de labs y examenes si están disponibles
            labs_info = "No lleva laboratorio"
            if labs_data:
                labs_info = (
                    f"\n- Nombre: {labs_data.get('nombre_examen', 'NA')}[{labs_data.get('clave_examen', 'NA')}]"
                    f"\n- Costo: {labs_data.get('costo', 'NA')}"
                )

            # Construcción de info de Exámenes
            examenes_info = "No tiene certificación asociada"
            if examenes_data:
                examenes_info = (
                    f"\n- Nombre: {examenes_data.get('nombre_examen', 'NA')}[{examenes_data.get('clave_examen', 'NA')}]"
                    f"\n- Costo: {examenes_data.get('costo', 'NA')}"
                )
            disponibilidad = "Habilitado" if data.get('disponible', 'NA')== 'Si' and data.get('estatus_curso', '') == 'Liberado' \
                else "En habilitación" if data.get('disponible', 'NA') == 'No' and data.get('estatus_curso', '') != 'Liberado' \
                else "NA"
            
            try:                        
            #4. Construir la respuesta final con los datos ya corregidos
                     
                response_template = f""" 
                **Curso**
                - Clave: {data.get('clave', 'NA')}
                - Nombre: {data.get('nombre', 'NA')}V{data.get('version', 'NA')}
                - Tecnología/ Línea de negocio: {data.get('tecnologia_id', 'NA')} / {data.get('linea_negocio_id', 'NA')}
                - Entrega: {data.get('entrega', 'NA')}
                - Estatus del curso: {disponibilidad}
                - Tipo de curso: {data.get('tipo_curso_id', 'NA')}
                - Sesiones y Horas: {data.get('sesiones', 'NA')} / {data.get('horas', 'NA')}
                - Precio: {data.get('precio', 'NA')}{data.get('nombre_moneda', 'NA')}
                - Examen:\n{examenes_info}
                - Laboratorio:\n{labs_info}
                - País: {data.get('pais', 'NA')}
                - Complejidad: {data.get('complejidad_id', 'NA')}
                - Link temario: {link_temario}
                """
                
                #print(" Claves en data antes de formatear:", list(data.keys()))            
                formatted_response = response_template
                return formatted_response
            except Exception as e:
                print("Error en format_response:")
                traceback.print_exc()
                #return "error al formatear la respuesta"   
            return labs_data,examenes_data                
        
        if not isinstance(human_message, HumanMessage):
            logging.debug("Converting human_message to HumanMessage")
            human_message = HumanMessage(content=str(human_message))
 
        self.add_to_history(human_message)
        
        if conversation_history:
            context=" ".join([msg.content for msg in conversation_history[-5:]])
        else:
            context=""
        def ordenar_cursos(cursos, ns="Cursos"):
            comp_map = {'CORE': 1,'FOUNDATIONAL': 2,'BASIC': 3,'INTERMEDIATE': 4,'SPECIALIZED': 5,'ADVANCED': 6}
            tipo_map = {'Intensivo': 1, 'Digital': 2, 'Programa': 3}

            def key(doc):
                d = self.extract_data_from_text(doc.page_content, namespace=ns)
                disp = d.get("disponible", "No")
                comp = comp_map.get(d.get("complejidad_id", "ADVANCED"), 6)
                tipo = tipo_map.get(d.get("tipo_curso_id", "Programa"), 3)
                return (0 if disp == "Si" else 1, comp, tipo)

            return sorted(cursos, key=key)
        
        def claves_coinciden(curso: dict, otro: dict) -> bool:
            c1 = (curso.get("clave") or "").strip()
            c2 = (otro.get("clave") or "").strip()
            if not c1 or not c2:
                return False
            if c1 == c2:
                return True
            # compara la “base” de la clave (antes de espacios o paréntesis)
            b1 = re.split(r"\s|\(", c1, 1)[0]
            b2 = re.split(r"\s|\(", c2, 1)[0]
            return b1 == b2
        
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
                    data = self.extract_data_from_text(doc.page_content, namespace="Cursos")
                    if not data.get("link_temario") or data["link_temario"] in ("NA", "Temario no encontrado"):
                        base = (data.get("clave") or "").split()[0]
                        t = temarios_idx.get(base)
                        if t:
                            data["link_temario"] = t.get("link_temario", "Temario no encontrado")
                    formatted_response = format_response(data)
                    print(formatted_response)

            if cursos_subcontratados:
                print("\n**En otras modalidades de entrega te ofrecemos:**")
                for curso in ordenar_cursos(cursos_subcontratados):
                    data = self.extract_data_from_text(doc.page_content, namespace="Cursos")
                    if not data.get("link_temario") or data["link_temario"] in ("NA", "Temario no encontrado"):
                        base = (data.get("clave") or "").split()[0]
                        t = temarios_idx.get(base)
                        if t:
                            data["link_temario"] = t.get("link_temario", "Temario no encontrado")
                    formatted_response = format_response(data)
                    print(formatted_response)

            if cursos_habilitacion:
                print("\n**Cursos en habilitación:**")
                for curso in ordenar_cursos(cursos_habilitacion):
                    data = self.extract_data_from_text(doc.page_content, namespace="Cursos")
                    if not data.get("link_temario") or data["link_temario"] in ("NA", "Temario no encontrado"):
                        base = (data.get("clave") or "").split()[0]
                        t = temarios_idx.get(base)
                        if t:
                            data["link_temario"] = t.get("link_temario", "Temario no encontrado")
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
            Primero, analiza la consulta del usuario contenida en "{human_message.content}", esta consulta la debes responder con la información de "Listado de cursos y temarios", la respuesta debe estar muy relacionada para responder a "{human_message.content}", sólo incluye los cursos que den respuesta a la consulta.  
            **Certificaciones**:  
               - Si la consulta incluye términos como "certificación" o "certificaciones", muestra **sólo** la lista de certificaciones disponibles en los cursos proporcionados en la lista Cursos#. **No incluyas ningún curso en el que el campo "Certificación" sea "Ninguna" o está vací­o**. 
               - Si la consulta es sobre si alguna certificación está disponible/vigente o no, responde y adicional, sugiere el nombre y clave de los cursos que están asociados a esa certificación en caso de que sí­ este disponible.
               - **Asegúrate de que cada certificación se mencione una sola vez**. Si varios cursos comparten la misma certificación, menciónala solo una vez.
               - **No incluyas ningún curso adicional**. 
               **Listado de cursos:**
            """
        leader_precio =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            Primero, analiza la consulta del usuario contenida en "{human_message.content}", esta consulta la debes responder con la información de "Listado de cursos y temarios", la respuesta debe estar muy relacionada para responder a "{human_message.content}", sólo incluye el o los cursos que den respuesta a la consulta.  
            **Precio**: 
               - Si la consulta incluye términos como "precio" o "costo" y menciona un curso especí­fico o una categorí­a de cursos (como "cursos de Python"),  proporciona **únicamente con el precio** del curso mencionado. En caso de ser más de un cruso, imprime una sola lí­nea por curso, mostrando el nombre del curso seguido de su precio. Ejemplo: "Curso Python Developer: 1095.0 USD". 
               - Si la consulta incluye términos como "precio" o "costo" y menciona un examen o laboratorio especí­fico ,  proporciona **únicamente con el costo** del examen o laboratorio mencionado. En caso de ser más de una solicitud, imprime una sola lí­nea por examen o laboratorio, mostrando el nombre del examen o laboratorio seguido de su precio. Ejemplo: "Laboratorios rentados directamente con Microsoft: 10.0 USD". 
               - No incluyas ninguna otra información sobre el curso como disponibilidad, sesiones, tecnologí­a, etc. Responde exclusivamente con el nombre del curso y su precio. Incluye al menos 5 cursos con su precio.
               **Listado de cursos** 
            """
        leader_general =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en espaÃ±ol.
            Primero, analiza la consulta del usuario contenida en "{human_message.content}", esta consulta la debes responder con la información de "Listado de cursos y temarios", la respuesta debe estar muy relacionada para responder a "{human_message.content}", sólo incluye los cursos que den respuesta a la consulta.  
            **Cursos en general y temas especí­ficos**: 
               - Si la consulta es sobre qué se le podrá recomendar a un cliente, responde con todas las entregas de un mismo curso. Por ejemplo si la pregunta es: "Mi cliente desea capacitarse en temas básicos de redes Cisco, Â¿qué curso le debo ofrecer?", la respuesta debe ser: CCNA, CCNA (Digital), CCNA (Digital-CLC).
               - Si el usuario sólo te saluda, salúdalo y ofrece tu ayuda de forma amable.
               - Si la consulta es sobre un tema, tecnología o familia específica como "{human_message.content}" (por ejemplo, cursos de Java o cursos de cyberseguridad ), responde comenzando con una introducción como "Estos son algunos cursos relacionados con tu consulta:" y luego lista los cursos de los proporcionados que correspondan a esa tecnologí­a, tema o familia. **Incluye todos los cursos que se relacionen con "{human_message.content}", y sigue estrictamente el formato completo proporcionado: **
               - Si el curso que muestras es subcontratado:sí­, muestra su información completa y añade al final el siguiente mensaje: "Sin embargo, al haber cursos subcontratados, te recomiendo ponerte en contacto con un Ing.preventa para más información". No importa si sólo es un curso subcontratado, muestra siempre el mensaje (por ejemplo, tenemos cursos de ISO 9001?) responde con la información del curso y añade el mensaje final.
               - Si la consulta es sobre servicios específicos como, AWS Lambda, Kubernetes, contenedores, etc. muestra los cursos relacionados y al final añade "Sin embargo te sugiero ahondar más con un Ing. preventa"
               - A partir de la información de el **Listado de Cursos y temarios:** responde en un parrafo concreto de manera muy humanizada la consulta del usuario, si son múltiples cursos haz un listado.
               - Al hacer el resumen de cada curso recomendado, debes usar la información de los **Temarios** y no de los **cursos** en un párrafo que contenga la información mas relevente, el resumen de cada curso debe tener al menos 150 palabras y por curso lo unico en negrilla es el nombre del curso y su clave entre corchetes.
               - Seguido del resumen colocaras la plantilla de los cursos correspondiente de **Cursos**
               **Listado de cursos** 
               """
        leader_agnostico=f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que están en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            Primero, analiza la consulta del usuario contenida en "{human_message.content}", esta consulta la debes responder con la información de "Listado de cursos y temarios", la respuesta debe estar muy relacionada para responder a "{human_message.content}", solo incluye los cursos que den respuesta a la consulta.
               - Si la consulta incluye "agnóstico" o "agnósticos", tu respuesta debe contener únicamente cursos que incluyan las palabras "Comptia", "CCC", "APMG" o "GKI" en cualquier parte de su información.
               - No muestres cursos que contengan palabras como "Microsoft", "AWS", "Cisco", "ECCouncil", o "Palo Alto" en cualquier parte de su información.
               - Al hacer el resumen de cada curso recomendado, debes usar la información de los **Temarios** y no de los **cursos** en un parrafo que contenga la información mas relevente. El resumen de cada curso debe tener al menos 150 palabras y por curso lo unico en negrilla es el nombre del curso y su clave entre corchetes.
               - Seguido del resumen colocaras la plantilla del curso correspondiente de **Cursos**:
            
            
            \n**Curso** [Course Key]  
            - **Nombre**: [nombre] V[version]
            - **Tecnologí­a/ Linea de negocio**: [Technology] / [linea_negocio_id] 
            - **Entrega**: [entrega]
            - **Estatus del curso**: [Availability] 
            - **Tipo de curso**: [Course Type]        
            - **Sesiones y Horas**: [Number of Sessions] / [Number of Hours]
            - **Precio**: [Price] [Currency] 
            - **Examen: [tipo_elemento_2]**
                -Clave: [exam key]
                -Precio: [cost]    
            - **Laboratorio: [Approach]** 
                -Clave: [exam key]
                -Precio: [cost]    
            - **País: [Country]**
            - **Complejidad**: [Complexity Level]    
            - **Link al temario**: [Link]               

               - Si algún campo no tiene información, usa "NA". Asegúrate de que "Link al temario" diga "Temario no encontrado" si el valor original es "NA".
               - Si no encuentras información de ningún curso agnóstico, muestra el mensaje: "Disculpa, no tengo esta información. Favor de ponerte en contacto con un Ing. preventa."
               **Listado de Cursos y temarios:**
            """
        leader_temarios =f"""Eres un asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            Primero, analiza la consulta del usuario contenida en "{human_message.content}", esta consulta la debes responder con la información de "Listado de cursos y temarios", la respuesta debe estar muy relacionada para responder a "{human_message.content}", sólo incluye los cursos que den respuesta a la consulta.
               - Responde en un parrafo concreto de manera muy humanizada la consulta del usuario, si son múltiples cursos haz un listado.
               **Listado de temarios:**
               - Si te piden el temario de un curso, proporcionale el enlace (link) de acceso con una breve descripción del mismo. Ejemplo: "Cuál es el temario del curso AZ-900T00: [OUTLINE]\nPuedes encontrar el temario en el siguiente enlace: https://sce.netec.com/files/ventas_consultivas_solicitudes/AZ-900T00%20Microsoft%20Azure%20Fundamentals1733357572.pdf"
            """
        leader_recomendacion2 =f"""Eres un asistente útil,tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
            Primero, analiza la consulta del usuario contenida en "{human_message.content}", esta consulta la debes responder con la información de "Listado de cursos y temarios", la respuesta debe estar muy relacionada para responder a "{human_message.content}", sólo incluye los cursos que den respuesta a la consulta.
               - A partir de la información de el **Listado de Cursos y temarios:** responde en un párrafo concreto de manera muy humanizada la consulta del usuario, si son múltiples cursos haz un listado.
               - Ordena los cursos en dos niveles: primero por complejidad y luego por tipo de curso. Usa el siguiente criterio para la ordenación:
                Complejidad (de menor a mayor): 'CORE' (1), 'FOUNDATIONAL' (2), 'BASIC' (3), 'INTERMEDIATE' (4), 'SPECIALIZED' (5) y al final 'ADVANCED' (6).
                Tipo de curso (de menor a mayor): 'Intensivo' (1), 'Digital' (2), 'Programa' (3).
                Si dos cursos tienen la misma complejidad, usa el tipo de curso como criterio secundario de ordenación.
               - Si la consulta menciona un rubro en especí­fico como "versión" asegurate de responder solo con esa información, por ejemplo "¿cuál es la versión del curso CISM?" responde "La versión del curso CISM es 2024"
               - Muestra siempre primero los cursos habilitados, después, como recomendacion adicional, los cursos subcontratados al final los cursos en 'Habilitación'.
               - Si el curso es subcontratado, sólo muestra la siguiente información:
            \n**Curso** [Course Key]  
            - **Nombre**: [nombre]
            - **Información adicional**:https://netec.sharepoint.com/Subcontrataciones/subcontratacioneslatam/SitePages/Inicio.aspx
               - Al hacer el resumen de cada curso recomendado, debes usar la información de los **Temarios** y no de los **cursos** en un párrafo que contenga la información mas relevente, el resumen de cada curso debe tener al menos 150 palabras y por curso lo único en negrilla es el nombre del curso y su clave entre corchetes.
               - Seguido del resumen colocaras la plantilla del curso correspondiente de **Cursos**:                      
            
        
            \n**Curso** [Course Key]  
            - **Nombre**: [nombre] V[version]
            - **Tecnologí­a/ Linea de negocio**: [Technology] / [linea_negocio_id] 
            - **Entrega**: [entrega]
            - **Estatus del curso**: [Availability] 
            - **Tipo de curso**: [Course Type]        
            - **Sesiones y Horas**: [Number of Sessions] / [Number of Hours]
            - **Precio**: [Price] [Currency] 
            - **Examen: [tipo_elemento_2]**
                -Clave: [exam key]
                -Precio: [cost]
            - **Laboratorio: [Approach]**
                -Clave: [exam key]
                -Precio: [cost]        
            - **País: [Country]**
            - **Complejidad**: [Complexity Level]    
            - **Link al temario**: [Link]
             
               - Si la consulta menciona un rubro en especí­fico como "versión" asegurate de responder solo con esa información, por ejemplo "¿cuál es la versión del curso CISM?" responde "La versión del curso CISM es 2024"
               - Si existen varios cursos con una "clave" similar, incluye todos los cursos de esa "clave". (Por ejemplo, "Clave: CCNA", "Clave: CCNA (Digital)", "Clave: CCNA (Digital-CLC) En este caso la respuesta debe incluir todos los cursos con las diferentes entregas: CCNA, CCNA (Digital), CCNA (Digital-CLC)).
               - Utiliza el formato completo proporcionado para el curso, y asegúrate de que "Link al temario" diga "Temario no encontrado" si el valor original es "NA" y que cada dato, se muestre en forma de lista.
               - Si la consulta es sobre un tema que no conoces como "cursos confirmados", "tutorías", "desarrollos"; muestra el siguiente mensaje "Disculpa, esta es información con la que no cuento, favor de ponerte en contacto con un Ing. Preventa"
            **Listado de Cursos y temarios:**
            """
        leader_labs =f"""Eres un asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
                - Cuando te pregunten por el laboratorio de un curso, responde con el nombre del laboratorio y su clave entre corchetes, en caso de que el curso cuente con uno. (Por ejemplo: - ¿El curso Microsoft Azure Security Technologies tiene laboratorio? - Sí, el curso tiene el laboratorio: Laboratorio rentado directamente con Microsoft [Lab-MIC-5]
                - Si te preguntan por el la clave de un laboratorio, toma la información de 'labs_data' . Por ejemplo (- ¿Cuál es la clave del laboratorio del curso Certified Vulnerability Assessor? - La clave del laboratorio es Lab-Mile2) Recupera esta información de 'labs_data' 
                - **Responde sólo lo que te pregunten.**
                
                """
        leader_exa=f"""Eres un asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la información en español.
                - Cuando te pregunten por el examen de un curso, responde con el nombre del examen y su clave entre corchetes, en caso de que el curso cuente con uno. (Por ejemplo: -¿El curso A+ (HExt) tiene examen? - Sí, para este curso encuentro dos exámanes: certificación [220-1101 or 220-1102] y certificación [Examen Preparacion N4S]
                - Recupera la información de nombre en  'examenes_data'
                - No es lo mismo examen de certificación que examen de curso, si el usuario pregunta por el examen de un curso, responde con el examen de curso, no con el de certificación.                         
                - **Responde sólo lo que te pregunten.**
                """
        leader_No =f"""Eres una asistente útil, tu nombre es Max y proporcionas información de los cursos que estan en "Listado de cursos y temarios".
            Siempre debes mostrar la informaciónn en español.
                - Si el usuario sólo te saluda, salúdalo y ofrece tu ayuda de forma amable.
                - Cuando no quede clara la consulta, debes responder con: ¡Hola! Parece que tu consulta no está clara o no está relacionada con nuestros temas. ¿Podrías darnos más detalles o especificar qué información estás buscando?
                - No cuentas con la informaciónn del tipo: "cursos confirmados", "tutorí­as", "desarrollos". En esos casos, responde con : Disculpa, esa es información con la que no cuento. Favor de ponerte en contacto con un Ing. Preventa
                """
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
            namespace="Laboratorios"
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
        cleaned = clean_query(human_message.content)
        enhanced_query = f"{cleaned}"
        #enhanced_query = f"{clean_query(human_message.content)}"  
        if not enhanced_query.strip():
            print("Consulta vací­a, se omite la búsqueda en Pinecone.")
            return "Disculpa, no puedo procesar una consulta vaci­a. Por favor, intenta nuevamente."

        # Realizar la búsqueda solo si enhanced_query no estÃ¡ vacÃ­o
        try:
            metadata_filter = {"pais": user_country} if user_country else None
            documents = self.retriever(namespace=namespace, top_k=40).get_relevant_documents(query=enhanced_query,metadata_filter=metadata_filter)
            #documents = self.retriever(namespace=namespace, top_k=40, country=user_country).get_relevant_documents(query=enhanced_query)
            #Postfiltrado
            documents = documents if namespace == "Temarios" else ([d for d in documents if d.metadata.get("pais") == user_country] or documents)
            print(f"Documentos recuperados en {namespace}: {len(documents)}")
            dump_docs(namespace, documents, max_docs=None)  # None = imprime TODOS
            save_docs(namespace, documents)
            documents = sorted(documents, key=lambda doc: doc.metadata.get("orden", 0))

        
        except Exception as e:
            print(f"Error al realizar la búsqueda en Pinecone: {e}")
            logging.error(f"Error al realizar la búsqueda en Pinecone: {e}")
            return "Disculpa, hubo un problema al realizar la búsqueda. Intenta nuevamente."
        
        #Realiza búsquedas adicionales en 'labs' y 'examenes si la intención es 'cursos'
        labs_documents = []
        exams_documents = []
        if intencion in ["Cursos","Laboratorios","Examenes"]:
            #additional_namespace_2 = "Labs" if namespace == "Cursos" else "Examenes"
            try:
                metadata_filter = {"pais": user_country} if user_country else None
                labs_documents = self.retriever(namespace='Laboratorios', top_k=40).get_relevant_documents(query=enhanced_query,metadata_filter=metadata_filter)
                exams_documents = self.retriever(namespace='Examenes', top_k=40).get_relevant_documents(query=enhanced_query,metadata_filter=metadata_filter)
                dump_docs("Laboratorios", labs_documents, max_docs=None)
                dump_docs("Examenes", exams_documents, max_docs=None)
                save_docs("Laboratorios", labs_documents)
                save_docs("Examenes", exams_documents)
                #labs_documents=self.retriever(namespace='Laboratorios', top_k=40).get_relevant_documents(query=enhanced_query)
                #exams_documents=self.retriever(namespace='Examenes', top_k=40).get_relevant_documents(query=enhanced_query)
                
            except Exception as e:
                print(f'Error al realizar la búsqueda en Pinecone: {e}')
                return 'Disculpa, hubo un problema al realizar la búsqueda. Intenta nuevamente'
        # Si no se encontraron exámenes, se hace una búsqueda alternativa con el nombre del curso 
            if not exams_documents:
                print("No se encontraron exámenes en la primera búsqueda. Intentando búsqueda alternativa")
                try:                
                    metadata_filter = {"pais": {"$eq": user_country}}
                    exams_documents=self.retriever(namespace='Examenes').get_relevant_documents(query=data.get('nombre', ''),metadata_filter=metadata_filter)
                    #exams_documents=self.retriever(namespace='Examenes', country=user_country).get_relevant_documents(query=data.get('nombre', ''))
            
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

            additional_documents = self.retriever(namespace=additional_namespace, top_k=40).get_relevant_documents(query=enhanced_query, **({} if additional_namespace == "Temarios" else {"metadata_filter": metadata_filter}))
            dump_docs(additional_namespace, additional_documents, max_docs=None)
            save_docs(additional_namespace, additional_documents)
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
                additional_doc_temarios = self.retriever(namespace="Cursos", top_k=40).get_relevant_documents(query=claves_string,metadata_filter=metadata_filter)
            # Inserta los documentos adicionales al principio
                documents = additional_doc_temarios + documents
            documents_cursos = documents
            temarios_idx = {}
            if intencion in ("Recomendacion", "Cursos", "Agnostico"):
                tem_docs = self.retriever(namespace="Temarios", top_k=40).get_relevant_documents(
                    query=enhanced_query
                )
                dump_docs("Temarios", tem_docs, max_docs=None)
                save_docs("Temarios", tem_docs)
                temarios_idx = self._index_temarios(tem_docs)
            # Combina los resultados de ambos `namespace`
            #documents.extend(additional_documents)  
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
        debug_save_prompt("formatted_documents1", formatted_documents)
        # Procesar cursos directos
        formatted_documents.append("\n**Cursos Disponibles:**")
        debug_save_prompt("formatted_documents2", formatted_documents)
        labs_data = {}
        examenes_data = {}

        for doc in sorted_direct_courses:
            try:
                # usa lc_id si viene en metadata, si no, usa el texto del doc (tu SELECT en orden)
                lc_id_val = doc.metadata.get('lc_id') or (doc.page_content or "")
                data = self.extract_data_from_text(doc.page_content, namespace="Cursos")
                if not data.get("link_temario") or data["link_temario"] in ("NA", "Temario no encontrado"):
                    base = (data.get("clave") or "").split()[0]
                    t = temarios_idx.get(base)
                    if t:
                        data["link_temario"] = t.get("link_temario", "Temario no encontrado")

                labs_data = next(
                    (
                        self.extract_data_from_text(d.page_content, namespace="Laboratorios")
                        for d in labs_documents
                        if claves_coinciden(
                            data,
                            self.extract_data_from_text(d.page_content, namespace="Laboratorios"),
                        )
                    ),
                    {},
                )

                examenes_data = next(
                    (
                        self.extract_data_from_text(d.page_content, namespace="Examenes")
                        for d in exams_documents
                        if claves_coinciden(
                            data,
                            self.extract_data_from_text(d.page_content, namespace="Examenes"),
                        )
                    ),
                    {},
                )

                formatted_documents.append(format_response(data, labs_data, examenes_data))
            except Exception as e:
                print(f"Error procesando curso directo: {e}")

        # Procesar cursos subcontratados
        if sorted_subcontracted_courses:
            formatted_documents.append("\n**En otras modalidades de entrega te ofrecemos:**")
            for doc in sorted_subcontracted_courses:
                if 'tokens' in doc.metadata:
                    try:
                        data = self.extract_data_from_text(doc.page_content, namespace="Cursos")
                        if not data.get("link_temario") or data["link_temario"] in ("NA", "Temario no encontrado"):
                            base = (data.get("clave") or "").split()[0]
                            t = temarios_idx.get(base)
                            if t:
                                data["link_temario"] = t.get("link_temario", "Temario no encontrado")
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
                            # tras calcular lab/exam por matching…
                            if not examenes_data:
                                try:
                                    exams_alt = self.retriever(namespace='Examenes', top_k=40).get_relevant_documents(
                                        query=f"{data.get('clave','')} {data.get('nombre','')}",
                                        metadata_filter={"pais": user_country} if user_country else None
                                    )
                                    examenes_data = next(
                                        (self.extract_data_from_text(d.page_content, "Examenes")
                                        for d in exams_alt
                                        if claves_coinciden(data, self.extract_data_from_text(d.page_content, "Examenes"))),
                                        {}
                                    )
                                except Exception as e:
                                    print(f"[Fallback Examenes] {e}")

                            if not labs_data:
                                try:
                                    labs_alt = self.retriever(namespace='Laboratorios', top_k=40).get_relevant_documents(
                                        query=f"{data.get('clave','')} {data.get('nombre','')}",
                                        metadata_filter={"pais": user_country} if user_country else None
                                    )
                                    labs_data = next(
                                        (self.extract_data_from_text(d.page_content, "Laboratorios")
                                        for d in labs_alt
                                        if claves_coinciden(data, self.extract_data_from_text(d.page_content, "Laboratorios"))),
                                        {}
                                    )
                                except Exception as e:
                                    print(f"[Fallback Labs] {e}")

                            # Agregar la plantilla formateada
                            formatted_documents.append(format_response(data, labs_data, examenes_data))

                    except Exception as e:
                        print(f"Error procesando curso subcontratado: {e}")
        debug_save_prompt("formatted_docs_count", f"{len(formatted_documents)} items")
        debug_save_prompt("leader_usado", leader)
        document_texts=[doc for doc in formatted_documents]      
        debug_save_prompt("document_texts", document_texts)
        history_content = self.format_history()  # Obtener el historial formateado
        system_message_content = f"{leader}{'. '.join(map(str,document_texts))}" #f"{leader1}{claves_text}{'. '.join(document_texts)}"
        system_message = SystemMessage(content=system_message_content)
        debug_save_prompt("system_message_COMPLETA", system_message_content)
 
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
                debug_save_prompt("system_message_COMPLETA2", system_message)
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

        
        log_entry={
                "user_name":user_name or "desconocido",
                "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "question":human_message.content,
                "response":response_content
            }
        log_interaction(**log_entry)

        def handle_user_login(user_name: str) -> str:
            pais = get_user_country(user_name)
            if pais:
                return f"¡Hola {user_name}! Veo que eres de {pais}."
            else:
                return f"No he podido encontrar el país para el usuario {user_name}."

        reply = handle_user_login(user_name)
        st.write(reply)
        print(get_user_country(user_name))
