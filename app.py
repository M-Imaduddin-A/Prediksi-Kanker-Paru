import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt # Import baru untuk plotting
import seaborn as sns          # Import baru untuk plotting
# from PIL import Image # Tidak diperlukan lagi jika plot dibuat langsung

# --- 1. Memuat Model yang Sudah Disimpan ---
try:
    # Ganti nama file .pkl jika Anda sudah menyimpan dengan nama 'best_...'
    log_reg_model = joblib.load('best_logistic_regression_resampled_tuned_model.pkl')
    rf_model = joblib.load('best_random_forest_resampled_tuned_model.pkl')
    st.sidebar.success("Model berhasil dimuat!")
except FileNotFoundError:
    st.sidebar.error("Error: File model tidak ditemukan. Pastikan 'best_logistic_regression_resampled_tuned_model.pkl' dan 'best_random_forest_resampled_tuned_model.pkl' ada di direktori yang sama.")
    st.stop() # Menghentikan aplikasi jika model tidak ditemukan

# --- 2. Judul dan Deskripsi Aplikasi ---
st.title("Prediksi Risiko Kanker Paru")
st.write("""
Aplikasi ini memprediksi risiko kanker paru berdasarkan gejala dan faktor risiko yang Anda masukkan.
Ini adalah alat demonstrasi dan tidak boleh digunakan sebagai pengganti diagnosis medis profesional.
""")

# --- 3. Sidebar untuk Pemilihan Model ---
st.sidebar.header("Pengaturan Model")
selected_model = st.sidebar.selectbox(
    "Pilih Model untuk Prediksi:",
    ("Random Forest", "Regresi Logistik")
)

# --- 4. Input Pengguna ---
st.header("Masukkan Informasi Anda:")

# Input GENDER (0=M, 1=F)
gender_input = st.radio("Jenis Kelamin", ("Laki-laki", "Perempuan"))
gender_encoded = 0 if gender_input == "Laki-laki" else 1

# Input AGE
age_input = st.slider("Usia", min_value=1, max_value=100, value=40)

# Fungsi untuk membuat input biner (1=Yes, 0=No)
def binary_input(label):
    option = st.radio(label, ("Ya", "Tidak"))
    return 1 if option == "Ya" else 0

st.subheader("Gejala dan Faktor Risiko:")
# Inputs untuk fitur biner (pastikan sesuai urutan kolom saat training model)
# Urutan kolom saat training: ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
# Perhatikan spasi di nama kolom 'FATIGUE ', 'ALLERGY '
smoking = binary_input("Merokok?")
yellow_fingers = binary_input("Jari Kuning?")
anxiety = binary_input("Sering Cemas?")
peer_pressure = binary_input("Tekanan Teman Sebaya?")
chronic_disease = binary_input("Memiliki Penyakit Kronis?")
fatigue = binary_input("Sering Merasa Lelah?")
allergy = binary_input("Memiliki Alergi?")
wheezing = binary_input("Mengalami Mengi (Wheezing)?")
alcohol_consuming = binary_input("Mengonsumsi Alkohol?")
coughing = binary_input("Sering Batuk?")
shortness_of_breath = binary_input("Mengalami Sesak Napas?")
swallowing_difficulty = binary_input("Kesulitan Menelan?")
chest_pain = binary_input("Mengalami Nyeri Dada?")

# --- 5. Mengumpulkan Input ke dalam DataFrame untuk Prediksi ---
# Pastikan urutan dan nama kolom sama persis dengan X_train saat model dilatih!
input_data = pd.DataFrame([[
    gender_encoded,
    age_input,
    smoking,
    yellow_fingers,
    anxiety,
    peer_pressure,
    chronic_disease,
    fatigue, # Perhatikan spasi jika ada di nama kolom asli
    allergy, # Perhatikan spasi jika ada di nama kolom asli
    wheezing,
    alcohol_consuming,
    coughing,
    shortness_of_breath,
    swallowing_difficulty,
    chest_pain
]], columns=[
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', # Pastikan spasi di sini sesuai dengan nama kolom asli
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'
])

# --- 6. Membuat Prediksi ---
st.subheader("Hasil Prediksi:")

if st.button("Prediksi Risiko Kanker Paru"):
    model_to_use = None
    if selected_model == "Random Forest":
        model_to_use = rf_model
    elif selected_model == "Regresi Logistik":
        model_to_use = log_reg_model

    if model_to_use:
        prediction = model_to_use.predict(input_data)[0]
        prediction_proba = model_to_use.predict_proba(input_data)[0]

        st.write(f"Model yang digunakan: **{selected_model}**")

        if prediction == 1:
            st.error("### Risiko Kanker Paru: YA")
            st.write(f"Berdasarkan input Anda, model memprediksi **kemungkinan tinggi** kanker paru.")
        else:
            st.success("### Risiko Kanker Paru: TIDAK")
            st.write(f"Berdasarkan input Anda, model memprediksi **kemungkinan rendah** kanker paru.")

        st.write(f"Probabilitas Prediksi (No Cancer / Cancer): **{prediction_proba[0]:.2f} / {prediction_proba[1]:.2f}**")
        st.info("Catatan: Ini adalah prediksi dari model. Selalu konsultasikan dengan profesional medis untuk diagnosis dan saran.")
    else:
        st.warning("Silakan pilih model untuk membuat prediksi.")

# --- 7. Perbandingan Performa Model ---
st.title("Perbandingan Performa Model Prediksi Kanker Paru")
st.write("Berikut adalah perbandingan metrik performa antara Model Regresi Logistik dan Random Forest (setelah tuning dan resampling).")

# HARDCODED PERFORMANCE METRICS (sesuaikan dengan hasil training Anda yang sebenarnya)
# Ini diambil dari output terakhir yang Anda berikan untuk model yang dituning dan di-resample
metrics_data = {
    'Model': ['Regresi Logistik', 'Random Forest'],
    # Metrik untuk kelas '1' (Cancer)
    'Accuracy': [0.9500, 0.9667],
    'Precision': [0.9500, 0.9623],
    'Recall': [1.0000, 1.0000],
    'F1-Score': [0.9744, 0.9808],
    'ROC-AUC': [0.9850, 0.9950] # Ini adalah nilai placeholder, ganti dengan nilai ROC-AUC aktual jika Anda memilikinya
}
df_performance = pd.DataFrame(metrics_data).set_index('Model')

# Tampilkan tabel performa
st.subheader("Tabel Perbandingan Performa Model")
st.dataframe(df_performance.round(4))

# Membuat dan menampilkan grafik perbandingan performa
st.subheader("Grafik Perbandingan Performa Model")

df_plot = df_performance.stack().reset_index()
df_plot.columns = ['Model', 'Metric', 'Value']

# Menggunakan st.pyplot() untuk menampilkan figure Matplotlib
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='Metric', y='Value', hue='Model', data=df_plot, palette='viridis', ax=ax)
ax.set_ylim(0, 1.0) # Metrik performa biasanya antara 0 dan 1
ax.set_title('Perbandingan Performa Model (Regresi Logistik vs Random Forest)')
ax.set_ylabel('Nilai Metrik')
ax.set_xlabel('Metrik')
ax.legend(title='Model')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

st.pyplot(fig) # Tampilkan plot di Streamlit

# Hapus bagian try-except Image.open karena grafik dibuat langsung
# try:
#     img = Image.open('model_performance_comparison.png')
#     st.image(img, caption='Grafik Perbandingan Performa Model')
# except FileNotFoundError:
#     st.error("File grafik 'model_performance_comparison.png' tidak ditemukan. Pastikan sudah dibuat.")
