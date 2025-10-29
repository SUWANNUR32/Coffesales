import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Coffee Sales Prediction", layout="centered")

# --- Header Aplikasi ---
st.title("☕ Prediksi Penjualan Kopi")
st.markdown("""
Aplikasi ini memprediksi nilai penjualan (**money**) berdasarkan waktu penjualan:
- 🕒 **Jam transaksi**
- 📅 **Urutan hari**
- 📆 **Urutan bulan**

Gunakan aplikasi ini untuk memantau dan memprediksi performa penjualan kopi berdasarkan waktu tertentu.
""")

# --- Input Data ---
st.header("📥 Masukkan Data Waktu Penjualan")
hour = st.number_input("Jam transaksi (0–23):", min_value=0, max_value=23, value=10)
weekday_sort = st.number_input("Urutan hari (1–7):", min_value=1, max_value=7, value=3)
month_sort = st.number_input("Urutan bulan (1–12):", min_value=1, max_value=12, value=5)

# --- Prediksi ---
if st.button("🔮 Prediksi Penjualan"):
    try:
        # Load model & scaler
        model = joblib.load("model_rf_coffee.joblib")
        scaler = joblib.load("scaler_coffee.joblib")

        # Buat DataFrame input baru
        new_data = pd.DataFrame({
            "hour_of_day": [hour],
            "Weekdaysort": [weekday_sort],
            "Monthsort": [month_sort]
        })

        # Transformasi data & prediksi
        scaled = scaler.transform(new_data)
        pred = model.predict(scaled)[0]

        # --- Hasil Prediksi Utama ---
        st.success(f"💰 **Prediksi Penjualan (money): ${pred:,.2f}**")

        # --- Visualisasi Prediksi Tunggal ---
        st.subheader("📊 Grafik Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Prediksi"], [pred], color="skyblue")
        ax.set_ylabel("Nilai Penjualan (Money)")
        ax.set_title("Prediksi Penjualan Tunggal")
        st.pyplot(fig)

        # --- Simulasi Tren per Jam (0–23) ---
        st.subheader("📈 Simulasi Tren Penjualan Berdasarkan Jam")
        hours = np.arange(0, 24)
        sim_data = pd.DataFrame({
            "hour_of_day": hours,
            "Weekdaysort": [weekday_sort]*24,
            "Monthsort": [month_sort]*24
        })
        sim_scaled = scaler.transform(sim_data)
        sim_preds = model.predict(sim_scaled)

        # Gabungkan hasil prediksi ke DataFrame
        hasil_simulasi = pd.DataFrame({
            "Jam": hours,
            "Prediksi_Penjualan": sim_preds
        })

        # --- Tampilkan Data Hasil Prediksi ---
        st.write("📋 **Tabel Hasil Prediksi per Jam:**")
        st.dataframe(hasil_simulasi.style.format({"Prediksi_Penjualan": "{:,.2f}"}))

        # --- Grafik Tren Prediksi per Jam ---
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(hours, sim_preds, marker='o', color='orange', linewidth=2)
        ax2.set_xlabel("Jam Transaksi")
        ax2.set_ylabel("Prediksi Penjualan (Money)")
        ax2.set_title("Simulasi Tren Penjualan per Jam")
        st.pyplot(fig2)

        # --- Statistik Ringkas ---
        st.subheader("📈 Statistik Ringkas Prediksi Harian")
        st.write(f"• Rata-rata Penjualan per Jam: **${np.mean(sim_preds):,.2f}**")
        st.write(f"• Penjualan Tertinggi: **${np.max(sim_preds):,.2f}** (Jam ke-{np.argmax(sim_preds)})")
        st.write(f"• Penjualan Terendah: **${np.min(sim_preds):,.2f}** (Jam ke-{np.argmin(sim_preds)})")

    except FileNotFoundError:
        st.error("⚠️ File model atau scaler tidak ditemukan. Pastikan 'model_rf_coffee.joblib' dan 'scaler_coffee.joblib' ada di folder repo.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.markdown("---")
st.caption("Dibuat oleh: **Suwannur32** | Project: Coffee Sales Prediction ☕")
