import os
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# ✅ Set page config at the top
st.set_page_config(page_title="Fabriconator", layout="centered")

# ✅ Sidebar title and upload
st.sidebar.title("🧵 Fabriconator")
uploaded_file = st.sidebar.file_uploader("Upload a fabric image", type=["jpg", "jpeg", "png"])

# ✅ Description
st.title("Fabric Defect Detection")
st.write("""
**Fabriconator** is an AI-powered tool that detects defects in fabric images.  
It classifies images into one of the following categories:  
👉 `Good`, `Hole`, `Line`, or `Spot`.
""")

# ✅ Load model and labels
@st.cache_resource
def load_model_and_labels():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "keras_Model.h5")
    labels_path = os.path.join(base_dir, "labels.txt")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    if not os.path.exists(labels_path):
        st.error(f"❌ Labels file not found at: {labels_path}")
        st.stop()

    model = load_model(model_path, custom_objects={"DepthwiseConv2D": DepthwiseConv2D}, compile=False)
    labels = [line.strip() for line in open(labels_path)]
    return model, labels

model, class_names = load_model_and_labels()

# ✅ Prediction
def predict(image: Image.Image):
    # NOTE: Teachable Machine uses 224x224 by default — change only if your model doesn't expect resizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1  # Normalization used by Teachable Machine

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)[0]
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[index])
    return class_name, confidence

# ✅ On upload
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fabric Image", use_container_width=True)

    class_name, confidence = predict(image)

    st.success(f"✅ **Prediction:** {class_name}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}")
