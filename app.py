from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os

app = Flask(__name__)

# ✅ Load model terlatih (.h5)
MODEL_PATH = 'model_jagung.h5'
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model '{MODEL_PATH}' berhasil dimuat.")
    else:
        print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan di {os.getcwd()}")
except Exception as e:
    print(f"❌ Error saat memuat model: {e}")
    model = None

# ✅ Ganti urutan ini sesuai hasil print(train_ds.class_names) dari notebook pelatihan Anda!
# Ini adalah bagian KRUSIAL yang harus cocok persis dengan urutan yang dipelajari model.
class_names = [
    'Bercak Daun (Gray Leaf Spot)', # Contoh: Ini harus kelas indeks 0 jika model memprediksi indeks 0
    'Hawar Daun (Northern Leaf Blight)', # Contoh: Ini harus kelas indeks 1
    'Karat Daun (Rust)', # Contoh: Ini harus kelas indeks 2
    'Sehat (Healthy)' # Contoh: Ini harus kelas indeks 3
]
print(f"✅ Class names yang digunakan di app.py: {class_names}")

# ✅ Fungsi preprocessing gambar agar sesuai dengan training
def preprocess_image(image):
    print("\n--- Memulai Preprocessing Gambar ---")
    print(f"Ukuran gambar asli (PIL): {image.size}, Mode: {image.mode}") # PIL.Image size (width, height)

    # Resize gambar ke ukuran input model (pastikan 256, 256 sesuai dengan model Anda)
    target_size = (256, 256)
    image = image.resize(target_size)
    print(f"Gambar di-resize menjadi: {image.size}")
    
    # Ubah gambar ke numpy array
    image_array = np.array(image)
    print(f"Shape array setelah resize: {image_array.shape}, Dtype: {image_array.dtype}")
    
    # Pastikan gambar memiliki 3 channel (RGB), bahkan jika aslinya grayscale
    if image_array.ndim == 2: # Jika gambar grayscale (hanya 2 dimensi: tinggi, lebar)
        image_array = np.stack((image_array,)*3, axis=-1) # Duplikat channel untuk menjadi RGB
        print(f"Gambar grayscale diubah ke RGB. Shape baru: {image_array.shape}")
    elif image_array.shape[-1] == 4: # Jika gambar memiliki 4 channel (RGBA)
        image_array = image_array[..., :3] # Ambil hanya 3 channel pertama (RGB)
        print(f"Gambar RGBA diubah ke RGB. Shape baru: {image_array.shape}")

    # Normalisasi piksel ke rentang 0-1
    # Jika model Anda menggunakan jenis normalisasi lain (misalnya ImageDataGenerator rescale=1./255 atau preprocess_input dari aplikasi Keras),
    # pastikan ini cocok!
    image_array = image_array.astype("float32") / 255.0
    print(f"Shape array setelah normalisasi: {image_array.shape}, Dtype: {image_array.dtype}")
    print(f"Min pixel value setelah normalisasi: {np.min(image_array):.4f}")
    print(f"Max pixel value setelah normalisasi: {np.max(image_array):.4f}")

    # Tambahkan dimensi batch (1, 256, 256, 3)
    image_array = np.expand_dims(image_array, axis=0)
    print(f"Shape array setelah expand_dims (siap untuk model): {image_array.shape}")
    print("--- Preprocessing Gambar Selesai ---\n")
    return image_array

@app.route('/')
def index():
    print("Permintaan diterima untuk halaman utama (index.html).")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Menerima permintaan prediksi...")
    if model is None:
        print("❌ Error: Model belum dimuat. Tidak dapat melakukan prediksi.")
        return redirect(url_for('index'))

    if 'image' not in request.files:
        print("❌ Tidak ada file 'image' dalam request.")
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        print("❌ Nama file kosong.")
        return redirect(url_for('index'))

    try:
        # Membuka file gambar sebagai RGB
        image = Image.open(file.stream).convert("RGB")
        print("Gambar berhasil dibuka dan dikonversi ke RGB.")

        # Preprocessing gambar
        input_image = preprocess_image(image)
        
        # Melakukan prediksi
        print("Melakukan prediksi dengan model...")
        predictions = model.predict(input_image)
        print(f"Hasil prediksi mentah dari model: {predictions}")
        
        # Mendapatkan kelas dan keyakinan
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        
        # Pastikan confidence adalah float bawaan Python
        confidence = float(np.max(predictions) * 100.0) 

        print(f"✅ Prediksi selesai: Kelas '{predicted_class}' dengan keyakinan {confidence:.2f}%")
        
        # Mengirim data ke result.html
        return render_template('result.html', 
                               predicted_class=predicted_class, 
                               confidence=confidence)

    except UnidentifiedImageError:
        print("❌ Error: File yang diunggah bukan gambar yang dikenali.")
        return redirect(url_for('index'))
    except Exception as e:
        print(f"❌ Terjadi error tak terduga saat prediksi: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    print("Memulai server Flask...")
    app.run(debug=True)
