import streamlit as st
import pandas as pd
import joblib

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Coffee Sales Prediction", layout="centered")

# --- Header Aplikasi ---
st.title("☕ Prediksi Penjualan Kopi")
st.write("Masukkan data waktu penjualan untuk memprediksi nilai penjualan (**money**).")

# --- Input user ---
hour = st.number_input("Jam transaksi (0–23):", min_value=0, max_value=23, value=10)
weekday_sort = st.number_input("Urutan hari (1–7):", min_value=1, max_value=7, value=3)
month_sort = st.number_input("Urutan bulan (1–12):", min_value=1, max_value=12, value=5)

# --- Jalankan prediksi ---
if st.button("🔮 Prediksi Penjualan"):
    try:
        # Load model dan scaler (pastikan nama file sesuai)
        model = joblib.load("rf_model.joblib")
        scaler = joblib.load("scaler_coffee.joblib")

        # Buat DataFrame dari input user
        new_data = pd.DataFrame({
            "hour_of_day": [hour],
            "Weekdaysort": [weekday_sort],
            "Monthsort": [month_sort]
        })

        # Scaling dan prediksi
        scaled = scaler.transform(new_data)
        prediction = model.predict(scaled)[0]

        # Hasil prediksi
        st.success(f"💰 **Prediksi Penjualan: ${prediction:,.2f}**")

    except FileNotFoundError:
        st.error("⚠️ File 'rf_model.joblib' atau 'scaler_coffee.joblib' tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau melakukan prediksi: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Dibuat oleh: **Suwannur32** | Project Coffee Sales Prediction ☕")
