import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Coffee Sales Prediction", layout="centered")

# --- Header Aplikasi ---
st.title("☕ Prediksi Penjualan Kopi")
st.markdown("""
Aplikasi ini memprediksi nilai penjualan (**money**) berdasarkan waktu penjualan:
- 🕒 **Jam transaksi**
- 📅 **Urutan hari**
- 📆 **Urutan bulan**

Tambahan visualisasi: **Feature Importance**, **Tren per Jam**, **Tren per Bulan**, dan **Golden Hour**.
""")

# --- Fungsi pembantu formatting uang ---
def money_fmt(x, pos):
    """Formatter untuk axis (menambahkan $ dan pemisah ribuan)."""
    return f'${x:,.0f}'
money_formatter = FuncFormatter(money_fmt)

# --- Input Data ---
st.header("📥 Masukkan Data Waktu Penjualan")
col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Jam transaksi (0–23):", min_value=0, max_value=23, value=10, step=1)
with col2:
    weekday_sort = st.number_input("Urutan hari (1–7):", min_value=1, max_value=7, value=3, step=1)
with col3:
    month_sort = st.number_input("Urutan bulan (1–12):", min_value=1, max_value=12, value=5, step=1)

st.markdown("---")

# --- Load model & scaler (sesuaikan nama file di repo) ---
MODEL_PATH = "rf_model.joblib"
SCALER_PATH = "scaler_coffee.joblib"

def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as fe:
        st.error(f"⚠️ File model atau scaler tidak ditemukan. Pastikan `{MODEL_PATH}` dan `{SCALER_PATH}` ada di folder repo.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model/scaler: {e}")
        st.stop()

model, scaler = load_artifacts()

# --- Prediksi ketika tombol ditekan ---
if st.button("🔮 Prediksi Penjualan"):
    try:
        new_data = pd.DataFrame({
            "hour_of_day": [hour],
            "Weekdaysort": [weekday_sort],
            "Monthsort": [month_sort]
        })

        scaled = scaler.transform(new_data)
        pred = model.predict(scaled)[0]

        st.success(f"💰 **Prediksi Penjualan (money): ${pred:,.2f}**")

        # Single bar chart
        st.subheader("📊 Grafik Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Prediksi"], [pred])
        axe. y-axis.set_major_formatter(money_formatter)
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

        hasil_simulasi = pd.DataFrame({
            "Jam": hours,
            "Prediksi_Penjualan": sim_preds
        })

        st.write("📋 **Tabel Hasil Prediksi per Jam:**")
        st.dataframe(hasil_simulasi.style.format({"Prediksi_Penjualan": "{:,.2f}"}))

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(hours, sim_preds, marker='o', linewidth=2)
        ax2.set_xlabel("Jam Transaksi")
        ax2.set_ylabel("Prediksi Penjualan (Money)")
        ax2.yaxis.set_major_formatter(money_formatter)
        ax2.set_title("Simulasi Tren Penjualan per Jam")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

        # --- Statistik Ringkas ---
        st.subheader("📈 Statistik Ringkas Prediksi Harian")
        st.write(f"• Rata-rata Penjualan per Jam: **${np.mean(sim_preds):,.2f}**")
        st.write(f"• Penjualan Tertinggi: **${np.max(sim_preds):,.2f}** (Jam ke-{int(np.argmax(sim_preds))}:00)")
        st.write(f"• Penjualan Terendah: **${np.min(sim_preds):,.2f}** (Jam ke-{int(np.argmin(sim_preds))}:00)")

        # --- Feature Importance (jika model punya atribut ini) ---
        st.subheader("🌟 Pengaruh Fitur Terhadap Prediksi (Feature Importance)")
        try:
            feature_names = ["hour_of_day", "Weekdaysort", "Monthsort"]
            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
            ax_imp.barh(feature_names, importances)
            ax_imp.set_xlabel("Importance Score")
            ax_imp.set_title("Feature Importance - Random Forest")
            st.pyplot(fig_imp)

            # Tabel ringkas importance
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)
            st.table(fi_df.style.format({"importance": "{:.4f}"}))
        except Exception:
            st.info("Model tidak menyediakan atribut `feature_importances_` (bukan model tree-based).")

        # --- Simulasi Tren Bulanan (1-12) ---
        st.subheader("📆 Tren Penjualan Berdasarkan Bulan (Simulasi)")
        months = np.arange(1, 13)
        sim_month_data = pd.DataFrame({
            "hour_of_day": [hour]*12,
            "Weekdaysort": [weekday_sort]*12,
            "Monthsort": months
        })
        sim_month_scaled = scaler.transform(sim_month_data)
        sim_month_pred = model.predict(sim_month_scaled)

        fig_month, ax_month = plt.subplots(figsize=(8, 4))
        ax_month.plot(months, sim_month_pred, marker="o", linewidth=2)
        ax_month.set_xlabel("Bulan")
        ax_month.set_ylabel("Prediksi Penjualan (Money)")
        ax_month.yaxis.set_major_formatter(money_formatter)
        ax_month.set_xticks(months)
        ax_month.set_title("Simulasi Tren Penjualan per Bulan")
        ax_month.grid(alpha=0.2)
        st.pyplot(fig_month)

        # --- Golden Hour (jam terbaik) ---
        st.subheader("⭐ Jam Emas Penjualan (Golden Hour)")
        best_hour = int(np.argmax(sim_preds))
        best_value = float(np.max(sim_preds))
        st.success(f"Jam terbaik: **{best_hour}:00** — estimasi penjualan **${best_value:,.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.markdown("---")
st.caption("Dibuat oleh: **Suwannur32** | Project: Coffee Sales Prediction ☕")
