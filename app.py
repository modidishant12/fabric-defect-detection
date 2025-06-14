import os
import streamlit as st
from keras.models import load_model

# Silence TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 👇 This should be first Streamlit command
st.set_page_config(page_title="Fabriconator", layout="centered")

st.sidebar.title("🧵 Fabriconator")
st.sidebar.markdown("Upload a fabric image to detect defects.")

# 🔽 Debug current directory
st.sidebar.write("📁 Files in app folder:")
st.sidebar.write(os.listdir("."))  # Show files for verification

# ✅ Model + labels loader
def load_model_and_labels():
    model_path = "keras_Model.h5"
    label_path = "labels.txt"

    # ✅ Extra logging
    if not os.path.exists(model_path):
        st.error("❌ keras_Model.h5 not found. Check filename and directory.")
        st.stop()

    if not os.path.exists(label_path):
        st.error("❌ labels.txt not found.")
        st.stop()

    from keras.layers import DepthwiseConv2D  # Needed for Teachable Machine models
    model = load_model(model_path, compile=False)
    labels = [line.strip() for line in open(label_path).readlines()]
    return model, labels

# ✅ Load
model, class_names = load_model_and_labels()
