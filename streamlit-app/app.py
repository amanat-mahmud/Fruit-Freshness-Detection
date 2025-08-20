import streamlit as st
from PIL import Image 
from model_helper import predict

st.set_page_config(page_title="Fruit Freshness Detection", page_icon="🍓", layout="centered")
st.title("Fruit Freshness Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB") 
    st.image(image, caption="Uploaded File", use_container_width=True)
    prediction = predict(image)                    

    st.info(f"Predicted Class: {prediction}")

