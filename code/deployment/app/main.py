import streamlit as st
from PIL import Image
import io
import requests
import os

API_URL = os.getenv("API_URL")

st.title("Butterfly Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "img"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded image', use_column_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes.seek(0)

    if st.button("Classify"):
        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        response = requests.post(API_URL, files=files)
        response_json = response.json()
        if response.status_code == 200:
            st.success("Image sent to API")
            st.write(f"Prediction: {response_json.get('Prediction')}")
        else:
            st.error(f"Error: {response.status_code}")