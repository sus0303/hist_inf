import os
import streamlit as st
import base64
from openai import OpenAI
import openai
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

Expert=" "
profile_imgenh=" "

# Inicializar session_state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'full_response' not in st.session_state:
    st.session_state.full_response = ""
if 'base64_image' not in st.session_state:
    st.session_state.base64_image = ""

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except FileNotFoundError:
        return "Error: La imagen no se encontró en la ruta especificada."


# ---------------- UI ----------------
st.set_page_config(page_title='Detector de Emociones')
st.title('🎭 Detector de Emociones')

with st.sidebar:
    st.subheader("Acerca de:")
    st.subheader("Dibuja una cara (feliz, triste, enojada, etc.) y la IA interpretará la emoción y te dará recomendaciones")

st.subheader("Dibuja una cara con una emoción y presiona analizar")

# Canvas
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider('Selecciona el ancho de línea', 1, 30, 5)
stroke_color = "#000000"
bg_color = '#FFFFFF'

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
ke = st.text_input('Ingresa tu Clave', type="password")
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=api_key)

analyze_button = st.button("Analiza la emoción", type="secondary")

# ---------------- PROCESAMIENTO ----------------
if canvas_result.image_data is not None and api_key and analyze_button:

    with st.spinner("Analizando ..."):
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8')).convert('RGBA')
        input_image.save('img.png')

        base64_image = encode_image_to_base64("img.png")
        st.session_state.base64_image = base64_image

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
                st.session_state.full_response = response.choices[0].message.content
                st.session_state.analysis_done = True

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- RESULTADO ----------------
if st.session_state.analysis_done:
    st.divider()
    st.subheader("Resultado ｡𖦹°‧")

    resultado = st.session_state.full_response
    st.write(resultado)

    # Feedback visual
    if "feliz" in resultado.lower():
        st.success("•ᴗ• Emoción positiva detectada")
    elif "triste" in resultado.lower():
        st.info("• ᴖ • Tómate un momento para ti")
    elif "enojado" in resultado.lower():
        st.warning("•̀ ᴖ •́  Libera esa energía de forma sana")
    elif "sorprendido" in resultado.lower():
        st.info(" ˶°ㅁ° Algo llamó tu atención")
    else:
        st.info(" - .•Emoción detectada")

# ---------------- WARNING ----------------
if not api_key:
    st.warning("Por favor ingresa tu API key.")
