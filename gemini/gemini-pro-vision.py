import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import io
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)    
    image = Image.open(io.BytesIO(picture.getvalue()))
    model = genai.GenerativeModel(model_name="gemini-pro-vision")
    response = model.generate_content(["how old is this person?take you best guess", image])
    message(response.text) 