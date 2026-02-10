import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(
    page_title="Plant Image Recognition",
    layout="centered"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #f5fff7;
}
h1 {
    color: #1b5e20;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- LOAD EXCEL ----------------
plants_df = pd.read_excel("plants_datas.xlsx")

# ---------------- NORMALIZATION FUNCTION ----------------
def normalize_name(name):
    return name.lower().replace("_", "").replace(" ", "").strip()

plants_df["normalized_name"] = plants_df["Plant Name"].apply(normalize_name)

# ---------------- UI ----------------
st.title("🌿 Plant Image Recognition System")
st.write("Upload a plant image to get its name, scientific name and benefits.")

uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    top_indices = preds.argsort()[-3:][::-1]
    top_confidences = preds[top_indices]
    top_class_names = [class_names[i] for i in top_indices]

    # ---------------- CONFIDENCE CHECK ----------------
    if top_confidences[0] < CONFIDENCE_THRESHOLD:
        st.error("❌ Plant not confidently recognized. Please upload a clearer image.")
    else:
        main_plant = top_class_names[0]
        confidence = top_confidences[0]

        normalized_model_name = normalize_name(main_plant)
        plant_row = plants_df[
            plants_df["normalized_name"] == normalized_model_name
        ]

        if not plant_row.empty:
            scientific_name = plant_row.iloc[0]["Scientific Name"]
            benefits = plant_row.iloc[0]["Benefits"]
        else:
            scientific_name = "Not available"
            benefits = "Not available"

        # ---------------- RESULT ----------------
        st.success("✅ Plant Identified Successfully!")
        st.markdown(f"🌱 **Plant Name:** {main_plant}")
        st.markdown(f"🔬 **Scientific Name:** {scientific_name}")
        st.markdown(f"💊 **Benefits:** {benefits}")
        st.markdown(f"📊 **Confidence:** {confidence * 100:.2f}%")

        # ---------------- TOP-3 BAR CHART ----------------
        st.subheader("📈 Top 3 Predictions")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(top_class_names[::-1], top_confidences[::-1], color="#66bb6a")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")

        for i, v in enumerate(top_confidences[::-1]):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center")

        st.pyplot(fig)
