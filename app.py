import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="☕ Coffee Sales Prediction", layout="centered")

st.title("📊 Prediksi Penjualan Coffee Shop ☕")
st.write("Masukkan data penjualan untuk memprediksi nilai 'money' ($).")

st.divider()

# Load model and scaler
try:
    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler_coffee.joblib")
    st.success("✅ Model & Scaler berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Tidak dapat memuat model: {e}")
    st.stop()

# INPUT FORM
st.header("📥 Input Prediksi")

col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=10)
with col2:
    weekday = st.number_input("Weekday Sort (1-7)", min_value=1, max_value=7, value=3)
with col3:
    month = st.number_input("Month Sort (1-12)", min_value=1, max_value=12, value=5)

if st.button("🔮 Prediksi Sekarang!"):
    try:
        # Format input
        new_data = pd.DataFrame([{
            "hour_of_day": hour,
            "Weekdaysort": weekday,
            "Monthsort": month
        }])

        # Standardize
        scaled = scaler.transform(new_data)
        scaled_df = pd.DataFrame(scaled, columns=new_data.columns)

        # Predict
        prediction = model.predict(scaled_df)[0]

        st.success(f"💰 Hasil Prediksi Penjualan: **${prediction:,.2f}**")

        # Visualization
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Prediksi Penjualan"], [prediction], color="orange")
        ax.set_ylabel("Money ($)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan: {e}")

st.markdown("---")
st.caption("✨ Dibuat oleh Suwannur32 | Powered by Streamlit & scikit-learn")
