"""
Streamlit App – Deteksi Penyakit Daun Jagung
Jalankan dengan perintah:
    streamlit run streamlit_app.py
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os

# ────────────────────────────────────────────────────────────────────────────────
# Konfigurasi model
# ────────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "model_jagung.h5"

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """Memuat model TensorFlow terlatih (format .h5)."""
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        return model
    st.error(f"Model '{path}' tidak ditemukan di {os.getcwd()}")
    return None

model = load_model(MODEL_PATH)

# Urutan nama kelas *harus* sama persis dengan saat pelatihan
class_names = [
    "Bercak Daun (Gray Leaf Spot)",
    "Hawar Daun (Northern Leaf Blight)",
    "Karat Daun (Rust)",
    "Sehat (Healthy)",
]

# ────────────────────────────────────────────────────────────────────────────────
# Fungsi pre‑processing gambar
# ────────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize → normalisasi [0–1] → tambahkan dimensi batch."""
    target_size = (256, 256)
    image = image.resize(target_size)

    image_array = np.array(image)
    # Pastikan 3 channel RGB
    if image_array.ndim == 2:  # grayscale → RGB
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[-1] == 4:  # RGBA → RGB
        image_array = image_array[..., :3]

    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # (1, 256, 256, 3)
    return image_array

# ────────────────────────────────────────────────────────────────────────────────
# UI Streamlit
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="centered",
)

st.title("🌽 Deteksi Penyakit Daun Jagung dengan CNN")

st.markdown(
    """
    Unggah foto daun jagung Anda kemudian klik **Prediksi**.
    
    Model akan mengklasifikasikan gambar ke salah satu kelas berikut:
    1. **Bercak Daun (Gray Leaf Spot)**
    2. **Hawar Daun (Northern Leaf Blight)**
    3. **Karat Daun (Rust)**
    4. **Sehat (Healthy)**
    """
)

uploaded_file = st.file_uploader(
    "Pilih gambar (.jpg / .jpeg / .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file and model:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        if st.button("🔍 Prediksi"):
            with st.spinner("Memproses gambar & memprediksi…"):
                input_image = preprocess_image(image)
                predictions = model.predict(input_image)

            predicted_class_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_class_index]
            confidence = float(np.max(predictions) * 100.0)

            st.success(f"**{predicted_class}**")
            st.info(f"Keyakinan model: **{confidence:.2f}%**")

    except UnidentifiedImageError:
        st.error("File yang diunggah bukan gambar yang valid.")

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.markdown(
    """
    **Model:** CNN TensorFlow (.h5)  
    **Ukuran input:** 256×256 piksel  
    **Framework:** Streamlit 1.x  
    **Developer:** Anda 😊
    """
)
