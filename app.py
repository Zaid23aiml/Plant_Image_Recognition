import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(page_title="Plant Image Recognition", layout="centered")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f5fff7;
}
h1 {
    color: #1b5e20;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>🌿 Plant Image Recognition System</h1>",
    unsafe_allow_html=True
)

st.write("Upload a plant image to get its name, scientific name, benefits and confidence.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model.h5")

model = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_excel("plants_data.xlsx")

plants_df = load_data()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- NORMALIZE ----------------
def normalize_name(name):
    return name.strip().lower().replace(" ", "_").replace("-", "_")

plants_df["Plant Name"] = plants_df["Plant Name"].apply(normalize_name)
class_names = [normalize_name(c) for c in class_names]

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📸 Upload Plant Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]

    # Top-3 predictions
    top_indices = preds.argsort()[-3:][::-1]
    top_names = [class_names[i] for i in top_indices]
    top_scores = [preds[i] * 100 for i in top_indices]

    max_confidence = top_scores[0]
    predicted_class = top_names[0]

    # ---------------- BAR CHART ----------------
    st.subheader("📊 Prediction Confidence (Top 3)")

    fig, ax = plt.subplots()
    ax.barh(top_names[::-1], top_scores[::-1])
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # ---------------- CONFIDENCE MESSAGE ----------------
    if max_confidence < CONFIDENCE_THRESHOLD * 100:
        st.warning(
            "⚠️ Low confidence prediction. "
            "Results may be inaccurate. Try a clearer image."
        )
    else:
        st.success("✅ High confidence prediction")

    # ---------------- PLANT DETAILS ----------------
    row = plants_df[plants_df["Plant Name"] == predicted_class]

    if not row.empty:
        plant_info = row.iloc[0]
        st.markdown(f"🌱 **Plant Name:** {predicted_class}")
        st.markdown(f"🔬 **Scientific Name:** {plant_info['Scientific Name']}")
        st.markdown(f"💊 **Benefits:** {plant_info['Benefits']}")
    else:
        st.warning("Plant details not found in database.")

    st.markdown(f"📈 **Top Confidence:** {max_confidence:.2f}%")
