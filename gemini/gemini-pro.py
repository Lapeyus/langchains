import os
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-pro")

st.title('Ask gemini')
question = st.text_input("Enter your question here:")

if st.button('Generate Answer'):
    # message(question, is_user=True)     
    response = model.generate_content(question)
    message(response.text) 
    # st.text(response.text)
