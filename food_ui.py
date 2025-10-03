import streamlit as st
from main import predict_image

st.title("Naija Food Classifier 🍲")
uploaded_file = st.file_uploader("Upload food image", type=["jpg", "png", "jfif"])
if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    result = predict_image("temp.jpg")
    st.success(f"Prediction: {result}")
  
