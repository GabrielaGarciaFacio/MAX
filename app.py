import streamlit as st
from pathlib import Path
from langchain.schema import HumanMessage
from models.hybrid_search_retreiver import HybridSearchRetriever
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

retriever = HybridSearchRetriever()

path = Path(__file__).parent.resolve()
hybrid_search_retreiver = HybridSearchRetriever()
st.set_page_config(page_title='MAX', layout='wide')


# Añade CSS personalizado con un método más directo
st.markdown(f"""
<style>
    .fixed-header {{
        position: fixed;
        top: 30px;
        width: 89%;
        background-color: #008389;
        z-index: 1000;
        padding: 5px 10px;
        box-shadow: 0 4px 2px -2px gray;
        text-align: left;
    }}
    .chat-container {{
        margin-top: 80px; 
        text-align: center;
    }}
    .main-content {{
        margin-top: 0px; 
    }}
    .stChatMessage {{
        margin-bottom: 10px;
    }}   
    
        
    .user-message {{
        background-color: #E0F0FF;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        font-size: 15px;
        font-weight: 500;
        color: #02419F;
    }}
    .bot-message {{
        background-color: #F0F0F0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        font-size: 15px;
        font-weight: 500;
        color: #333333;
    }}
</style>           
<div style="
    background-color:#00A1AF;     
    border-radius: 8px; 
    padding: 15px; 
    font-size: 25px; 
    font-weight: 500; 
    color: #FFFFFF; 
    margin-bottom: 0px;">
<h2>MAX</h2>        
</div>

<div style="
    background-color: #E0F0FF; 
    border-left: 5px solid #00A1AF; 
    border-radius: 8px; 
    padding: 15px; 
    font-size: 15px; 
    font-weight: 500; 
    color: #02419F; 
    margin-bottom: 20px;
    margin-top: 0px; /* Agrega espacio arriba para bajar el bloque */">
😊 <strong>¡Hola!</strong> Para brindarte una mejor respuesta, cuéntame más detalles. Incluye información como la complejidad del curso, el país, la tecnología o algún punto clave del temario. ¡Así podré ayudarte de forma más precisa!</div>
""", unsafe_allow_html=True)

# Asegúrate de que los estados necesarios están inicializados
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "context" not in st.session_state:
    st.session_state["context"] = ""

# Función para detectar si hay un cambio de tema en la consulta
def detect_topic_change(user_input):
    # Comparar con las consultas anteriores usando TF-IDF y Cosine Similarity
    conversation_history = [msg.content for msg in st.session_state["conversation_history"]]
    
    if len(conversation_history) > 0:
        vectorizer = TfidfVectorizer().fit_transform(conversation_history + [user_input])
        vectors = vectorizer.toarray()

        similarity_matrix = cosine_similarity(vectors)
        last_similarity = similarity_matrix[-1][:-1]  # Compara la última entrada con las anteriores

        # Si la similitud promedio es menor a 0.3, consideramos que es un nuevo tema
        if last_similarity.mean() < 0.0:
            return True
    return False

def handle_user_input(user_input):
    human_message = HumanMessage(content=user_input)
    
    # Detectar si hay un cambio de tema
    if detect_topic_change(user_input):
        # Si se detecta un cambio de tema, reiniciamos el contexto y el historial
        st.session_state["conversation_history"] = []
        st.session_state["context"] = "nuevo_tema"
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.success("Se ha detectado un nuevo tema. El historial ha sido reiniciado.")

    # Añade el mensaje del usuario al historial de conversación
    st.session_state.conversation_history.append(human_message)

    # Muestra primero el mensaje del usuario en un chat_message permanente
    #with st.chat_message("user"):
    st.markdown(f"<div class='user-message'><b>Tú: </b>{user_input}</div>", unsafe_allow_html=True)

    
    # Configura un contenedor para la respuesta en tiempo real
    #with st.chat_message("assistant"):
    response_placeholder = st.empty()
    response_content = ""  # Se inicializa una variable para acumular la respuesta completa

    try:
        # Transmisión de la respuesta en tiempo real
        for partial_response in retriever.rag(human_message, conversation_history=st.session_state.conversation_history):
            # Actualizar el contenido de la respuesta
            response_content = partial_response
            # Mostrar fragmentos en tiempo real en el mismo contenedor
            response_placeholder.markdown(f"<div class='bot-message'><b>Max: </b>{response_content}</div>", unsafe_allow_html=True)

    
        # Almacenar la respuesta completa en el historial
        st.session_state.generated.append(response_content)
        st.session_state.past.append(user_input)

    except Exception as e:
        st.error(f"Error durante la transmisión de respuesta: {e}")

def get_text():
    user_input = st.chat_input(placeholder="¿En qué puedo ayudarte hoy?", key="input")
    if user_input:
        with st.spinner(":robot_face: escribiendo..."):
            handle_user_input(user_input)
    return user_input

# Contenedor principal que contiene el campo de entrada del chat
with st.container():
    user_input = get_text()

st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Mostrar el historial de conversación anterior (mensajes previos)
if st.session_state["generated"] and len(st.session_state["generated"]) > 1:
    for i in range(len(st.session_state["past"]) - 1):
        
        st.markdown(f"<div class='user-message'><b>Tú: </b> {st.session_state['past'][i]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-message'><b>Max: </b>{st.session_state['generated'][i]}</div>", unsafe_allow_html=True)

    
