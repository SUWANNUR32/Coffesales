import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="☕ Coffee Sales Prediction Dashboard",
    page_icon="☕",
    layout="centered",
)

# --- Gaya CSS khusus ---
st.markdown("""
<style>
    body {
        background-color: #f8f4ef;
    }
    .main {
        background-color: #fff8f0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #4b3832;
    }
    .stButton>button {
        background-color: #6f4e37;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #8c5b3f;
        color: #fff;
    }
    .stSuccess {
        background-color: #d7ccc8;
        color: #3e2723;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📊 Prediksi Penjualan Coffee Shop")
st.markdown("Masukkan data waktu transaksi untuk memprediksi nilai **penjualan (money)** dalam satuan dolar ($).")

st.divider()

# --- Fungsi formatting uang ---
def money_fmt(x, pos):
    return f"${x:,.0f}"
money_formatter = FuncFormatter(money_fmt)

# --- Input Data ---
st.header("🕒 Input Data Waktu Penjualan")
col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Jam transaksi (0–23):", min_value=0, max_value=23, value=10)
with col2:
    weekday_sort = st.number_input("Urutan hari (1–7):", min_value=1, max_value=7, value=3)
with col3:
    month_sort = st.number_input("Urutan bulan (1–12):", min_value=1, max_value=12, value=5)

st.divider()

# --- Load model dan scaler ---
MODEL_PATH = "rf_model.joblib"
SCALER_PATH = "scaler_coffee.joblib"

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Tidak dapat memuat model: {e}")
        st.stop()

model, scaler = load_artifacts()

# --- Prediksi ---
if st.button("🔮 Prediksi Sekarang"):
    try:
        new_data = pd.DataFrame({
            "hour_of_day": [hour],
            "Weekdaysort": [weekday_sort],
            "Monthsort": [month_sort]
        })
        scaled = scaler.transform(new_data)
        pred = model.predict(scaled)[0]

        st.success(f"💰 **Prediksi Penjualan: ${pred:,.2f}**")

        # Grafik hasil prediksi
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Prediksi"], [pred], color="#6f4e37")
        ax.yaxis.set_major_formatter(money_formatter)
        ax.set_ylabel("Nilai Penjualan ($)")
        ax.set_title("Prediksi Penjualan Tunggal")
        st.pyplot(fig)

        # --- Simulasi Tren per Jam ---
        st.subheader("📈 Tren Penjualan per Jam (Simulasi)")
        hours = np.arange(0, 24)
        sim_data = pd.DataFrame({
            "hour_of_day": hours,
            "Weekdaysort": [weekday_sort]*24,
            "Monthsort": [month_sort]*24
        })
        sim_scaled = scaler.transform(sim_data)
        sim_preds = model.predict(sim_scaled)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(hours, sim_preds, marker='o', color="#8c5b3f", linewidth=2)
        ax2.set_xlabel("Jam Transaksi")
        ax2.set_ylabel("Prediksi Penjualan ($)")
        ax2.yaxis.set_major_formatter(money_formatter)
        ax2.set_title("Simulasi Tren Penjualan per Jam")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        # Statistik ringkas
        st.write(f"• Rata-rata Penjualan per Jam: **${np.mean(sim_preds):,.2f}**")
        st.write(f"• Penjualan Tertinggi: **${np.max(sim_preds):,.2f}** (Jam {int(np.argmax(sim_preds))}:00)")
        st.write(f"• Penjualan Terendah: **${np.min(sim_preds):,.2f}** (Jam {int(np.argmin(sim_preds))}:00)")

        # --- Feature Importance ---
        st.subheader("🌟 Feature Importance")
        try:
            feature_names = ["hour_of_day", "Weekdaysort", "Monthsort"]
            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
            ax_imp.barh(feature_names, importances, color="#a67b5b")
            ax_imp.set_xlabel("Importance Score")
            ax_imp.set_title("Pengaruh Fitur terhadap Prediksi")
            st.pyplot(fig_imp)
        except:
            st.info("Model ini tidak memiliki atribut `feature_importances_`.")

        # --- Golden Hour ---
        st.subheader("⭐ Jam Emas Penjualan")
        best_hour = int(np.argmax(sim_preds))
        best_value = float(np.max(sim_preds))
        st.success(f"Jam terbaik: **{best_hour}:00** — estimasi penjualan **${best_value:,.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.divider()
st.caption("☕ Dibuat oleh **Suwannur32** | Coffee Sales Prediction Dashboard")
