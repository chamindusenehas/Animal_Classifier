import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
import json
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Animal Classifier", page_icon="🐾", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('animal_classifier_model.h5')

@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()
class_labels = list(class_names.keys())

st.title("🐾 Animal Classifier")
st.markdown("""
Upload Your image and we will let you know what is the name of the animal!  
Supported animals:
            
Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra.
""")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.warning("Please upload an image.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(image, caption="Uploaded Image")



    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)


    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100


    with col2:
        tab1, tab2, tab3 = st.tabs(["Prediction", "top 3 Predictions"," Confidence"])
        with tab1:
            st.write(f"**{predicted_class}**")
        with tab2:
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            for idx in top_3_idx:
                label = class_labels[idx]
                conf = predictions[0][idx] * 100
                st.write(f"{label}: {conf:.2f}%")
        with tab3:
            scores = predictions[0] * 100
            fig = px.bar(x=class_labels, y=scores, labels={"x": "Animal", "y": "Confidence (%)"})
            st.plotly_chart(fig)
            