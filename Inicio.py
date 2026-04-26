import os
import streamlit as st
import base64
from openai import OpenAI
import openai
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# ---------------- CONFIG ----------------
st.set_page_config(page_title='Detector de Emociones', layout="centered")

# Fondo crema 🌿
st.markdown('<style>.stApp {background-color: #F5E9DA;}</style>', unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'full_response' not in st.session_state:
    st.session_state.full_response = ""
if 'base64_image' not in st.session_state:
    st.session_state.base64_image = ""

# ---------------- FUNCIONES ----------------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ---------------- UI ----------------
st.title('🎭 Dibuja tu emoción')

with st.sidebar:
    st.subheader("Acerca de:")
    st.write("Dibuja una cara (feliz, triste, enojada, etc.) y la IA interpretará tu emoción y te dará recomendaciones.")

st.subheader("Dibuja una cara con una emoción y presiona analizar")

# Canvas
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider('Ancho de línea', 1, 30, 5)
stroke_color = "#000000"
bg_color = "#FFFFFF"

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=300,
    width=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

# API KEY
ke = st.text_input('Ingresa tu API Key', type="password")
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=api_key)

# BOTÓN
analyze_button = st.button("🔍 Analizar emoción")

# ---------------- PROCESAMIENTO ----------------
if canvas_result.image_data is not None and api_key and analyze_button:

    with st.spinner("Analizando emoción..."):
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8')).convert('RGBA')
        input_image.save('img.png')

        base64_image = encode_image_to_base64("img.png")
        st.session_state.base64_image = base64_image

        # PROMPT NUEVO 🔥
        prompt_text = (
            "Analiza este dibujo de una cara o expresión y determina la emoción principal "
            "(feliz, triste, enojado, sorprendido, confundido, etc.). "
            "Responde en este formato:\n\n"
            "Emoción: <nombre>\n"
            "Descripción: <breve explicación>\n"
            "Recomendaciones:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ..."
        )

        try:
            full_response = ""
            message_placeholder = st.empty()

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

            if response.choices[0].message.content is not None:
                full_response += response.choices[0].message.content
                message_placeholder.markdown(full_response)

            st.session_state.full_response = full_response
            st.session_state.analysis_done = True

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- RESULTADO ----------------
if st.session_state.analysis_done:
    st.divider()
    st.subheader("🧠 Resultado del análisis")

    resultado = st.session_state.full_response
    st.write(resultado)

    # Feedback visual según emoción 🎭
    if "feliz" in resultado.lower():
        st.success("😊 ¡Se detecta una emoción positiva!")
    elif "triste" in resultado.lower():
        st.info("😢 Parece que necesitas un momento para ti")
    elif "enojado" in resultado.lower():
        st.warning("😡 Hay mucha energía, intenta liberarla sanamente")
    elif "sorprendido" in resultado.lower():
        st.info("😲 Algo te llamó la atención")
    else:
        st.info("🤔 Emoción interesante detectada")

# ---------------- WARNING ----------------
if not api_key:
    st.warning("Por favor ingresa tu API key.")
