import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="☕ Coffee Sales Prediction", layout="centered")

# --- Header ---
st.title("☕ Prediksi Penjualan Coffee Shop")
st.markdown("""
Aplikasi ini memprediksi **nilai penjualan (Money $)** berdasarkan:
- ⏱️ Jam pembelian
- 📅 Hari dalam minggu
- 🗓️ Bulan dalam tahun
""")

st.divider()

# --- Upload Dataset (Opsional) ---
st.subheader("📂 Upload Dataset (Opsional)")
uploaded_file = st.file_uploader("Unggah file dataset (.csv):", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📋 **5 Data Teratas:**")
    st.dataframe(df.head())

    if "money" in df.columns:
        st.subheader("📊 Distribusi Money")
        fig, ax = plt.subplots()
        df["money"].plot(kind='hist', bins=20, ax=ax)
        st.pyplot(fig)
else:
    st.info("Belum ada dataset — kamu tetap bisa prediksi di bawah 👇")

st.divider()

# --- Input untuk Prediksi ---
st.header("🧾 Input Data untuk Prediksi Penjualan")

col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Hour of Day (0–23):", min_value=0, max_value=23, value=10)
with col2:
    weekday = st.number_input("Weekday Sort (1–7):", min_value=1, max_value=7, value=3)
with col3:
    month = st.number_input("Month Sort (1–12):", min_value=1, max_value=12, value=5)

# --- Prediksi ---
if st.button("🔮 Prediksi Penjualan!"):
    try:
        # Load model & scaler
        model = joblib.load("rf_model.joblib")
        scaler = joblib.load("scaler_coffee.joblib")

        # Format ke DataFrame
        new_data = pd.DataFrame([{
            "hour_of_day": hour,
            "Weekdaysort": weekday,
            "Monthsort": month
        }])

        # Scaling fitur numerik
        scaled = scaler.transform(new_data)

        # Prediksi Penjualan
        prediction = model.predict(scaled)[0]

        st.success(f"💰 Prediksi Penjualan: **${prediction:,.2f}**")

        # Visualisasikan hasil
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar(["Prediksi Penjualan"], [prediction])
        ax2.set_ylabel("Money ($)")
        ax2.set_title("Hasil Prediksi Penjualan")
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("⚠️ Pastikan file 'rf_model.joblib' & 'scaler_coffee.joblib' tersedia di folder!")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Dibuat oleh: **Suwannur32** | Coffee Sales Prediction ☕ | Powered by Streamlit & scikit-learn")
