import streamlit as st
import pandas as pd
import joblib
import time

# Konfigurasi Halaman
st.set_page_config(page_title="Lead Scorer Pro", page_icon="🎯", layout="wide")

# Custom CSS untuk tampilan kartu hasil
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .score-card {
        background: #ffffff;
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 12px solid #007bff;
        margin-top: 20px;
    }
    .score-value { font-size: 70px; font-weight: 800; margin: 0; line-height: 1; }
    .status-text { font-size: 28px; font-weight: 700; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Load Model (Pastikan nama file sama dengan yang di bikin_model.py)
@st.cache_resource
def load_resources():
    # Menggunakan nama file yang sudah diperbaiki
    return joblib.load('best_lead_scoring_model.joblib')

try:
    data = load_resources()
    model = data['model']
    options = data['cat_options']
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan Anda sudah menjalankan 'bikin_model.py' dan file 'best_lead_scoring_model.joblib' ada di folder yang sama.")
    st.stop()

# Header Utama
st.title("🎯 Lead Conversion Intelligent Predictor")
st.write("Silakan lengkapi semua data di bawah ini untuk mendapatkan analisis skor.")
st.divider()

# Layout Utama dalam Satu Halaman (2 Kolom)
with st.container():
    col_input1, col_input2 = st.columns(2, gap="large")
    
    with col_input1:
        st.subheader("🌐 Interaksi Website")
        time_spent = st.slider("Total Waktu di Website (Detik)", 0, 2500, 500)
        visits = st.number_input("Total Kunjungan", min_value=0, value=2)
        pages = st.number_input("Rata-rata Halaman Dilihat", min_value=0.0, value=2.0)
        last_act = st.selectbox("Aktivitas Terakhir", options['Last Activity'])
        no_email = st.radio("Boleh di-email?", ["No", "Yes"], horizontal=True)

    with col_input2:
        st.subheader("👤 Profil & Sumber")
        occ = st.selectbox("Pekerjaan Saat Ini", options['What is your current occupation'])
        spec = st.selectbox("Bidang Spesialisasi", options['Specialization'])
        source = st.selectbox("Sumber Lead (Lead Source)", options['Lead Source'])
        origin = st.selectbox("Asal Lead (Lead Origin)", options['Lead Origin'])

# Tombol Analisis
st.write("") 
if st.button("🔥 ANALISIS PELUANG KONVERSI SEKARANG", use_container_width=True, type="primary"):
    # Siapkan Data
    input_dict = {
        'Total Time Spent on Website': time_spent,
        'TotalVisits': visits,
        'Page Views Per Visit': pages,
        'Last Activity': last_act,
        'Do Not Email': no_email,
        'What is your current occupation': occ,
        'Specialization': spec,
        'Lead Source': source,
        'Lead Origin': origin
    }
    
    with st.spinner("Sistem sedang melakukan perhitungan..."):
        time.sleep(1.2) 
        input_df = pd.DataFrame([input_dict])
        
        # Prediksi
        prob = model.predict_proba(input_df)[0][1]
        score = prob * 100
        formatted_score = "{:.2f}".format(score) # 2 angka belakang koma

        # Logika Warna dan Status
        if score >= 75:
            color, status, icon = "#28a745", "HOT LEAD", "🔥"
            msg = "Prioritas Utama! Leads ini memiliki minat yang sangat tinggi."
        elif score >= 40:
            color, status, icon = "#ffc107", "WARM LEAD", "⚡"
            msg = "Potensial. Kirimkan konten edukasi atau penawaran terbatas."
        else:
            color, status, icon = "#dc3545", "COLD LEAD", "❄️"
            msg = "Minat rendah. Masukkan ke dalam daftar pemeliharaan (nurturing)."

        # Tampilan Hasil (Kartu Skor)
        st.markdown(f"""
            <div class="score-card" style="border-top-color: {color};">
                <p style="color: #666; font-size: 18px; margin-bottom: 10px;">Probabilitas Konversi</p>
                <h1 class="score-value" style="color: {color};">{formatted_score}%</h1>
                <p class="status-text" style="color: {color};">{icon} {status} {icon}</p>
                <hr style="border: 0.5px solid #eee; margin: 20px 0;">
                <p style="color: #444; font-size: 16px;"><b>Saran:</b> {msg}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if score >= 75:
            st.balloons()