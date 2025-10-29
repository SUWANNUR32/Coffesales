import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="☕ Coffee Sales Prediction", layout="wide")

st.title("📊 Coffee Shop Sales Predictor ☕")
st.write("Prediksi nilai penjualan berdasarkan jam, hari, dan bulan transaksi.")

st.markdown("---")

# Load model & scaler
try:
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler_coffee.joblib")
    st.success("✅ Model & Scaler berhasil dimuat!")
except:
    st.error("❌ Gagal memuat model/scaler. Pastikan nama file benar dan ada di folder.")
    st.stop()

# Input User
st.header("🧾 Masukkan Data Untuk Prediksi")

col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Hour of Day (0–23):", 0, 23, 10)
with col2:
    weekday = st.number_input("Weekday Sort (1–7):", 1, 7, 3)
with col3:
    month = st.number_input("Month Sort (1–12):", 1, 12, 5)

if st.button("🔮 Prediksi Sekarang!"):
    try:
        new_data = pd.DataFrame([{
            "hour_of_day": hour,
            "Weekdaysort": weekday,
            "Monthsort": month
        }])

        st.write("📌 Data Input:")
        st.dataframe(new_data)

        # Scaling ✔ FIXED ✔
        scaled = scaler.transform(new_data)
        scaled_df = pd.DataFrame(scaled, columns=new_data.columns)

        prediction = model.predict(scaled_df)[0]

        st.success(f"💰 Prediksi Penjualan: **${prediction:,.2f}**")

        # Visualisasi
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Prediksi Money ($)"], [prediction])
        st.pyplot(fig)

        # Feature Importance
        st.subheader("📌 Feature Importance")
        importance = model.feature_importances_
        fig2, ax2 = plt.subplots()
        ax2.bar(new_data.columns, importance, color="orange")
        ax2.set_title("Pengaruh Fitur terhadap Penjualan")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.markdown("---")
st.caption("Dibuat oleh: **Suwannur32** | Coffee Sales Prediction ☕")
