import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import base64

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="♻ Smart Waste Classifier",
    page_icon="logo_icon.png",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #76c7c0, #f2f2f2);
        color: #000000;
    }
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #76c7c0, #ffffff);
    }
    .stApp {
        background: linear-gradient(180deg, #e0f7fa, #ffffff);
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: #ffffff;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Menu
# ----------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Upload Image", "Prediction Graphs", "Training History"])

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_waste_model():
    return load_model("waste_classifier.h5")

model = load_waste_model()

class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

# ----------------------------
# Preprocess & Predict
# ----------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_data
def predict_waste(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence, preds

# ----------------------------
# Helper: load images as base64 (for header bg)
# ----------------------------
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = "logo_icon.png"
title_bg_path = "header_bg.jpg"
img_base64 = get_base64_image(title_bg_path)

# ----------------------------
# App Header
# ----------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image(logo_path, width=80)
with col_title:
    st.markdown(
        f"""
        <div style="
            background-image: url('data:image/png;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            padding: 35px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.3);
        ">
            <h1 style='font-weight: bold; font-size: 36px; margin:0; font-family: "Segoe UI Emoji", "Apple Color Emoji", sans-serif;'>
                ♻ Smart Waste Classification App
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Upload Image Page
# ----------------------------
if menu == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("📸 Choose a waste image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state['uploaded_file'] = uploaded_file
        
        # Step 1: Show uploaded image first
        st.markdown("<h5 style='text-align:center;'>Uploaded Image</h5>", unsafe_allow_html=True)
        img_col1, img_col2, img_col3 = st.columns([1,3,1]) 
        with img_col2:
            st.image(img, use_container_width=True)
 

        # Step 2: Show button below image
        if st.button("Predict Waste", key="predict", use_container_width=True):
            predicted_class, confidence, preds = predict_waste(img)
            st.session_state['predicted_class'] = predicted_class
            st.session_state['confidence'] = confidence
            st.session_state['preds'] = preds

            # Step 3: Show results only after button click
            st.markdown(
                f"""
                <div style="
                    background-color: #d4edda;
                    padding: 12px;
                    border-radius: 10px;
                    width: 100%;
                    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
                    margin-top: 15px;
                    margin-bottom: 8px;
                ">
                    <p style='margin:0; font-weight:bold;'>🗑 Waste Type: {predicted_class.capitalize()}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    background-color: #d3d3d3;
                    padding: 12px;
                    border-radius: 10px;
                    width: 100%;
                    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
                ">
                    <p style='margin:0; font-weight:bold;'>✅ Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------
# Prediction Graphs Page
# ----------------------------
if menu == "Prediction Graphs":
    if 'uploaded_file' in st.session_state and 'preds' in st.session_state:
        st.subheader("Prediction Graph")
        img = Image.open(st.session_state['uploaded_file']).convert("RGB")
        preds = st.session_state['preds']
        pred_index = np.argmax(preds)

        st.markdown("<h5 style='text-align:center;'>Uploaded Image</h5>", unsafe_allow_html=True)
        img_col1, img_col2, img_col3 = st.columns([1,3,1])
        with img_col2:
            st.image(img, use_container_width=True)

        plt.style.use('ggplot')
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.plasma
        colors = [cmap(norm(p)) for p in preds]
        colors[pred_index] = (1.0, 0.5, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.barh(class_names, preds, color=colors, edgecolor='black', linewidth=1.5)
        ax.invert_yaxis()
        ax.set_title("Waste Prediction Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_xlim(0,1)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width*100:.1f}%', va='center', fontsize=10)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Probability Scale', rotation=270, labelpad=15)

        st.pyplot(fig)

    else:
        st.warning("⚠ Please upload and predict an image first in the 'Upload Image' section.")

# ----------------------------
# Training History Page
# ----------------------------
if menu == "Training History":
    st.subheader("Training Performance")
    try:
        with open("training_history.pkl", "rb") as f:
            history = pickle.load(f)

        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc)+1)

        fig2, ax = plt.subplots(1,2, figsize=(12,5))
        plt.style.use('seaborn-v0_8-darkgrid')

        ax[0].plot(epochs, acc, "bo-", label="Training Acc")
        ax[0].plot(epochs, val_acc, "ro-", label="Validation Acc")
        ax[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        ax[1].plot(epochs, loss, "bo-", label="Training Loss")
        ax[1].plot(epochs, val_loss, "ro-", label="Validation Loss")
        ax[1].set_title("Model Loss", fontsize=14, fontweight="bold")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        st.pyplot(fig2)

    except FileNotFoundError:
        st.warning("⚠ Training history not found. Save it as 'training_history.pkl'.")