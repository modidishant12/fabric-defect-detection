import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.layers import DepthwiseConv2D
import os

# ──────────────────────────────────────────────────────────────────────
# 🔧 Patch for Teachable Machine DepthwiseConv2D
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# ──────────────────────────────────────────────────────────────────────
# 📦 Load model and labels with caching
@st.cache_resource
def load_model_and_labels():
    model_path = "keras_Model.h5"
    labels_path = "labels.txt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        return None, None

    if not os.path.exists(labels_path):
        st.error(f"❌ Labels file not found at: {labels_path}")
        return None, None

    try:
        model = load_model(model_path, custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D}, compile=False)
        labels = [label.strip() for label in open(labels_path, "r").readlines()]
        return model, labels
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None

# ──────────────────────────────────────────────────────────────────────
# 🏠 Page setup
st.set_page_config(page_title="Fabriconator", page_icon="🧵")

# ──────────────────────────────────────────────────────────────────────
# 🎨 Sidebar UI
st.sidebar.title("🧵 Fabriconator")
st.sidebar.info("Upload an image of fabric to detect defects.")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ──────────────────────────────────────────────────────────────────────
# 📄 Main page info
st.title("🧠 Fabriconator - Fabric Defect Detection AI")
st.markdown("""
**Fabriconator** is an AI-powered image classification tool trained to detect **defects in fabric** using machine learning.
It can identify:
- 🕳️ Holes  
- 🔘 Spots  
- 📏 Lines  
- ✅ Good (No defect)

Upload a fabric image using the sidebar to get started!
""")

# ──────────────────────────────────────────────────────────────────────
# 🧠 Load model and labels
model, class_names = load_model_and_labels()

# ──────────────────────────────────────────────────────────────────────
# 📷 Handle uploaded image
if uploaded_file and model and class_names:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # 🧮 Convert image to numpy array and normalize (NO RESIZE)
        image_array = np.asarray(image)
        normalized_image = (image_array.astype(np.float32) / 127.5) - 1

        # 🧾 Ensure shape (1, h, w, 3)
        data = np.expand_dims(normalized_image, axis=0)

        # 🔍 Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # ✅ Display result
        st.success(f"🎯 **Prediction:** {class_name.strip()}")
        st.info(f"📊 **Confidence:** {confidence_score:.2%}")

    except Exception as e:
        st.error(f"❌ Failed to process image: {e}")
