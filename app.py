# coffe_sales_app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# 🔧 LOAD MODEL & SCALER
# ================================================================
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler_coffee.joblib")

# ================================================================
# 🎨 CONFIGURASI HALAMAN
# ================================================================
st.set_page_config(
    page_title="☕ Coffee Sales Predictor Dashboard",
    layout="wide",
    page_icon="☕"
)

# ================================================================
# 🧭 SIDEBAR
# ================================================================
st.sidebar.header("📊 Input Data Prediksi")
st.sidebar.write("Masukkan parameter berikut untuk memprediksi pendapatan penjualan kopi:")

hour = st.sidebar.slider("⏰ Jam Penjualan", 0, 23, 10)
weekday = st.sidebar.selectbox("📅 Hari ke-", [1, 2, 3, 4, 5, 6, 7], help="1 = Senin, 7 = Minggu")
month = st.sidebar.selectbox("🗓️ Bulan ke-", list(range(1, 13)), help="1 = Januari, 12 = Desember")

st.sidebar.markdown("---")
st.sidebar.info("Pastikan data input sesuai dengan pola penjualan historis agar prediksi lebih akurat.")

# ================================================================
# 🧠 PREDIKSI
# ================================================================
input_df = pd.DataFrame({
    "hour_of_day": [hour],
    "Weekdaysort": [weekday],
    "Monthsort": [month]
})

scaled_input = scaler.transform(input_df)
prediksi_money = model.predict(scaled_input)[0]

# ================================================================
# 📈 HALAMAN UTAMA
# ================================================================
st.title("☕ Coffee Sales Prediction Dashboard")
st.markdown("""
Dashboard ini memanfaatkan **Random Forest Regressor** untuk memprediksi pendapatan penjualan kopi
berdasarkan **jam, hari, dan bulan** transaksi.  
Gunakan panel di kiri untuk mengubah parameter input.
""")

# ================================================================
# 💰 HASIL PREDIKSI
# ================================================================
st.subheader("💵 Hasil Prediksi Pendapatan")

col1, col2 = st.columns([2, 3])

with col1:
    st.metric(
        label="Estimasi Penjualan (Money)",
        value=f"${prediksi_money:,.2f}",
        delta=None,
        help="Nilai ini adalah hasil prediksi pendapatan berdasarkan input Anda."
    )

with col2:
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediksi_money,
        title={'text': "Prediksi Penjualan"},
        gauge={'axis': {'range': [0, max(1000, prediksi_money * 1.5)]},
               'bar': {'color': "mediumseagreen"},
               'steps': [
                   {'range': [0, prediksi_money/2], 'color': "lightgray"},
                   {'range': [prediksi_money/2, prediksi_money], 'color': "lightgreen"}]
              }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# 🔍 INTERPRETASI
# ================================================================
st.markdown("### 📘 Interpretasi Hasil")
st.write(f"""
Berdasarkan input:
- Jam ke **{hour}**
- Hari ke **{weekday}**
- Bulan ke **{month}**

Model memperkirakan total pendapatan penjualan sekitar **${prediksi_money:,.2f}**.
""")

st.info("""
💡 **Tips Interpretasi:**
- Penjualan di jam sibuk (pagi dan sore) cenderung lebih tinggi.  
- Hari kerja biasanya memiliki pola berbeda dibanding akhir pekan.  
- Bulan-bulan liburan (Des–Jan) bisa memunculkan anomali positif.
""")

# ================================================================
# 📊 SIMULASI VISUAL (OPSIONAL)
# ================================================================
st.markdown("### 📉 Simulasi Perubahan Jam Penjualan")
jam_range = np.arange(0, 24)
simulasi_df = pd.DataFrame({
    "hour_of_day": jam_range,
    "Weekdaysort": weekday,
    "Monthsort": month
})

simulasi_scaled = scaler.transform(simulasi_df)
simulasi_df["Prediksi"] = model.predict(simulasi_scaled)

fig_sim = px.line(
    simulasi_df, x="hour_of_day", y="Prediksi",
    title="Perkiraan Pendapatan Berdasarkan Jam Penjualan",
    labels={"hour_of_day": "Jam", "Prediksi": "Pendapatan ($)"}
)
st.plotly_chart(fig_sim, use_container_width=True)

# ================================================================
# ⚙️ FOOTER
# ================================================================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit + Random Forest | @2025 Coffee Analytics Team")
