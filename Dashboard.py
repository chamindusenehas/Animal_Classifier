import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
from keras.applications.efficientnet import preprocess_input 
from PIL import Image
import numpy as np
import json
import plotly.graph_objects as go

def load_class_names(file_path='class_names.json'):
    try:
        with open(file_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        st.error("class_names.json file not found. Using default class names.")
        return [
            {"Bear": 0, "Bird": 1, "Cat": 2, "Cow": 3, "Deer": 4, "Dog": 5, "Dolphin": 6, "Elephant": 7, "Giraffe": 8, "Horse": 9, "Kangaroo": 10, "Lion": 11, "Panda": 12, "Tiger": 13, "Zebra": 14}
        ]

classes = load_class_names()
classes = list(classes.keys())


@st.cache_resource
def load_trained_model():
    return load_model('animal_classifier_model.keras')

model = load_trained_model()


st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMarkdown {
        color: #BDBDBD;
    }
    .uploadedFile {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45A049;
    }
    .stImage {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    .stSpinner {
        color: #4CAF50;
    }
    .css-1lcbmhc {
        background-color: #1A1A1A;
    }
    body {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stAlert {
        background-color: #2C2C2C;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Animal Classifier", layout="wide", initial_sidebar_state="expanded")


with st.sidebar:
    st.title("About the Model")
    st.markdown("""
    This is a high-precision animal image classifier trained on 15 categories using TensorFlow and Keras.
    
    Classes  
    {}
    
    Upload an image to classify it with confidence scores. The model achieves ~99% accuracy on the test set.
    """.format('\n- ' + '\n- '.join(classes)))
    st.markdown("---")
    st.caption("Built with using Streamlit")


st.title("Animal Image Classifier")
st.markdown("Upload an animal image to get instant classification results.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Classifying..."):
            try:
                image = load_img(uploaded_file, target_size=(224, 224))
                image_array = img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = preprocess_input(image_array)
                

                preds = model.predict(image_array)[0]
                pred_class_idx = np.argmax(preds)
                pred_class = classes[pred_class_idx]
                confidence = preds[pred_class_idx] * 100
                

                st.subheader("Prediction Result")
                st.success(f"""**Class:** {pred_class}  
                           **Confidence:** {confidence:.2f}%""")
                

                top_indices = np.argsort(preds)[-5:][::-1]
                top_classes = [classes[i] for i in top_indices]
                top_probs = [preds[i] * 100 for i in top_indices]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_probs,
                        y=top_classes,
                        orientation='h',
                        marker=dict(color='#4CAF50', line=dict(color='#FFFFFF', width=1)),
                        text=[f"{p:.2f}%" for p in top_probs],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Top 5 Predictions",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Class",
                    plot_bgcolor="#1E1E1E",
                    paper_bgcolor="#121212",
                    font=dict(color="#E0E0E0"),
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing the image: {str(e)}")

st.markdown("---")
st.caption("© 2025 Animal Classifier Dashboard")