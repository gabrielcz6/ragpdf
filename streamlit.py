import streamlit as st
import os
from streamlit_option_menu import option_menu
from utils.load_retriever import RetrieverLoader
from utils.utilschat import *
import base64
from io import BytesIO



# Función para convertir imágenes en Base64 a Bytes
def decode_base64_to_image(base64_str):
    image_bytes = base64.b64decode(base64_str)
    return BytesIO(image_bytes)

def reiniciarChat():
    """Función que reinicia el chat y borra el historial"""
    st.toast("CHAT INICIADO", icon='🤖')
    # Inicializamos el historial de chat
    if "messages" in st.session_state:
        st.session_state.messages = []

def cargar_retriever(pdf_nombre):
         
         loader = RetrieverLoader(pdf_nombre)
         loader.load_data()
         retriever = loader.get_retriever()  
         loader=""
         return retriever  


def traer_pdf_nombres():
    carpeta_pkl = "./pkl"
    return [archivo.replace(".pkl", "") for archivo in os.listdir(carpeta_pkl) if archivo.endswith(".pkl")]

def chat():
    st.header("Chat con tu PDF")
    nombres_pdf = traer_pdf_nombres()
    
    if "archivo_seleccionado" not in st.session_state:
        st.session_state.archivo_seleccionado = nombres_pdf[0] if nombres_pdf else None
    
    st.write(f"Has seleccionado: {st.session_state.archivo_seleccionado}")

    # Inicialización del historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar los mensajes anteriores en el chat
    with st.container():
        for message in st.session_state.messages:
            if message["role"] != "system":  # Omitimos los mensajes del sistema
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Si el mensaje contiene imágenes, las mostramos
                    if "images" in message:
                        for base64_str in message["images"]:
                            st.image(decode_base64_to_image(base64_str), width=400, caption="Imagen")

    # Campo para el mensaje del usuario
    prompt = st.chat_input("¿Qué quieres saber?")
    
    if prompt:
        # Mostrar mensaje del usuario
        st.chat_message("user").markdown(prompt)
        
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Definir el historial de mensajes a enviar a la función `responderquery`
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        # Llamar a la función `responderquery` con el historial de mensajes
        respuesta = responderquery(messages, st.session_state.retriever)

        # Mostrar respuesta del asistente en el chat
        with st.chat_message("assistant"):
            st.write(respuesta[0])
            if len(respuesta) > 1:  # Verificar si hay imágenes
                for base64_str in respuesta[1]:
                    st.image(decode_base64_to_image(base64_str), width=400, caption="Imagen")
        
        # Agregar respuesta del asistente al historial incluyendo imágenes si existen
        mensaje_asistente = {"role": "assistant", "content": respuesta[0]}
        if len(respuesta) > 1:
            mensaje_asistente["images"] = respuesta[1]  # Guardar imágenes en el historial
        
        st.session_state.messages.append(mensaje_asistente)
        




def app():
    st.title("Aplicación RAG multimodal")
    
    with st.sidebar:
        st.sidebar.image("caa.png", width=290)

        # Obtener nombres de los archivos PDF
        nombres_pdf = traer_pdf_nombres()
        archivo_seleccionado = st.sidebar.selectbox(
            "Selecciona un archivo:", nombres_pdf, index=0
        ) if nombres_pdf else None
        
        
        # Verificar si el usuario quiere cargar el archivo seleccionado
        if archivo_seleccionado:
            if "archivo_seleccionado" not in st.session_state:
                st.session_state.archivo_seleccionado = None
            
            if st.sidebar.button("Seleccionar"):
                if st.session_state.archivo_seleccionado != archivo_seleccionado:
                    st.session_state.archivo_seleccionado = archivo_seleccionado
          
                    with st.spinner("Cargando retriever..."):
                        st.session_state.retriever = cargar_retriever(archivo_seleccionado)

                else:
                    st.sidebar.warning("Ya está seleccionado este archivo.")
                    
            if st.sidebar.button("Reset"):
                st.rerun()
        # Mostrar el menú
        menu = option_menu(
            menu_title='Menu',
            options=['Chat'],
            icons=['chat'],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "10px", "background-color": "#0D47A1"},
                "icon": {"color": "#64B5F6", "font-size": "23px"},
                "nav-link": {
                    "color": "#BBDEFB", "font-size": "20px", "text-align": "left", 
                    "margin": "5px 0px", "padding": "10px", "--hover-color": "#42A5F5"
                },
                "nav-link-selected": {"background-color": "#1976D2", "padding": "10px"},
            }
        )
    
    if menu == "Chat":
        chat()

if __name__ == "__main__":
    app()
