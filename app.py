import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="StatistikaLab - Analisis Statistika Deskriptif",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file, sheet_name=None):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            if sheet_name:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            return None, "Format file tidak didukung."
        return df, None
    except Exception as e:
        return None, f"Gagal membaca file: {str(e)}"

def calculate_mode(series):
    valid_series = series.dropna()
    if valid_series.empty:
        return "N/A"
    modes = valid_series.mode()
    if len(modes) == 0 or len(modes) == len(valid_series):
        return "Tidak Ada Modus"
    elif len(modes) == 1:
        return f"{modes.iloc[0]:.4g}"
    else:
        return f"Multimodal ({', '.join([f'{m:.4g}' for m in modes[:3]])}{'...' if len(modes) > 3 else ''})"

def build_frequency_table(series, num_bins=None):
    valid_data = series.dropna()
    n = len(valid_data)
    if n == 0:
        return pd.DataFrame()
    if num_bins is None or num_bins <= 0:
        k = int(np.ceil(1 + 3.322 * np.log10(n))) if n > 1 else 1
    else:
        k = num_bins
    counts, bin_edges = np.histogram(valid_data, bins=k)
    interval_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
    midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    rel_freq = (counts / n) * 100
    cum_freq = np.cumsum(counts)
    return pd.DataFrame({
        "Kelas Interval": interval_labels,
        "Titik Tengah (Xi)": midpoints,
        "Frekuensi Absolut (f)": counts,
        "Frekuensi Relatif (%)": np.round(rel_freq, 2),
        "Frekuensi Kumulatif": cum_freq
    })

st.markdown('<p class="main-header">📊 StatistikaLab</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Aplikasi Analisis Statistika Deskriptif Interaktif | UTS Statistika - S1 Matematika</p>', unsafe_allow_html=True)
st.divider()

st.sidebar.header("📁 1. Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("Unggah File Dataset (CSV / Excel):", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    sheet_name = None
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                sheet_name = st.sidebar.selectbox("Pilih Sheet Excel:", sheet_names)
            else:
                sheet_name = sheet_names[0]
        except Exception as e:
            st.error(f"Gagal memuat sheet Excel: {e}")

    df_raw, error_msg = load_data(uploaded_file, sheet_name)
    if error_msg:
        st.error(error_msg)
        st.stop()

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.header("🛠️ 2. Penanganan Missing Values")
    missing_strategy = st.sidebar.radio(
        "Strategi Perhitungan:",
        ["Abaikan Nilai Kosong (Pairwise/Listwise Default)", "Hapus Baris yang Memiliki Missing Value"]
    )
    if missing_strategy == "Hapus Baris yang Memiliki Missing Value":
        df = df_raw.dropna().copy()
    else:
        df = df_raw.copy()

    st.sidebar.header("🎯 3. Pemilihan Variabel")
    if not numeric_cols:
        st.sidebar.warning("⚠️ Tidak ditemukan kolom numerik.")
        selected_num_cols = []
    else:
        selected_num_cols = st.sidebar.multiselect("Pilih Variabel Numerik:", options=numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])

    st.sidebar.header("⚙️ 4. Opsi Statistik")
    ddof_option = st.sidebar.selectbox("Jenis Varians & Standar Deviasi:", options=["Sampel (ddof=1) - Default", "Populasi (ddof=0)"])
    ddof_val = 1 if "Sampel" in ddof_option else 0
    decimal_places = st.sidebar.number_input("Jumlah Angka Desimal:", min_value=0, max_value=6, value=3)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Ringkasan Data", "📐 Perhitungan Statistik", "📊 Visualisasi", "💡 Interpretasi", "📥 Ekspor"
    ])

    with tab1:
        st.subheader("📌 Metadata Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nama File", uploaded_file.name)
        c2.metric("Baris (N)", len(df_raw))
        c3.metric("Kolom", df_raw.shape[1])
        c4.metric("Kolom Numerik", len(numeric_cols))
        st.divider()
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.dataframe(df, use_container_width=True)
        with col_r:
            missing_df = pd.DataFrame({
                'Kolom': df_raw.columns,
                'Jumlah Kosong': df_raw.isnull().sum().values,
                'Persentase (%)': np.round((df_raw.isnull().sum().values / len(df_raw)) * 100, 2)
            })
            st.dataframe(missing_df, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("📐 Tabel Ringkasan Statistika Deskriptif")
        if selected_num_cols:
            cb = st.columns(4)
            chk_n = cb[0].checkbox("N", value=True)
            chk_min = cb[0].checkbox("Min", value=True)
            chk_max = cb[0].checkbox("Max", value=True)
            chk_mean = cb[1].checkbox("Mean", value=True)
            chk_median = cb[1].checkbox("Median", value=True)
            chk_mode = cb[1].checkbox("Modus", value=True)
            chk_var = cb[2].checkbox("Varians", value=True)
            chk_std = cb[2].checkbox("Std Dev", value=True)
            chk_q1 = cb[3].checkbox("Q1", value=True)
            chk_q2 = cb[3].checkbox("Q2", value=True)
            chk_q3 = cb[3].checkbox("Q3", value=True)

            recs = []
            for col in selected_num_cols:
                s = df[col].dropna()
                n_v = len(s)
                r = {"Variabel": col}
                if chk_n: r["N Valid"] = n_v
                if chk_min: r["Min"] = s.min() if n_v > 0 else np.nan
                if chk_max: r["Max"] = s.max() if n_v > 0 else np.nan
                if chk_mean: r["Mean"] = s.mean() if n_v > 0 else np.nan
                if chk_median: r["Median"] = s.median() if n_v > 0 else np.nan
                if chk_mode: r["Modus"] = calculate_mode(df[col])
                if chk_var: r["Varians"] = s.var(ddof=ddof_val) if n_v > ddof_val else np.nan
                if chk_std: r["Std. Deviasi"] = s.std(ddof=ddof_val) if n_v > ddof_val else np.nan
                if chk_q1: r["Q1 (25%)"] = s.quantile(0.25) if n_v > 0 else np.nan
                if chk_q2: r["Q2 (50%)"] = s.quantile(0.50) if n_v > 0 else np.nan
                if chk_q3: r["Q3 (75%)"] = s.quantile(0.75) if n_v > 0 else np.nan
                recs.append(r)

            sum_df = pd.DataFrame(recs).set_index("Variabel")
            disp_df = sum_df.copy()
            for c in disp_df.select_dtypes(include=[np.number]).columns:
                disp_df[c] = disp_df[c].apply(lambda x: f"{x:.{decimal_places}f}" if pd.notnull(x) else "N/A")
            st.dataframe(disp_df, use_container_width=True)

    with tab3:
        st.subheader("📊 Visualisasi Interaktif")
        if selected_num_cols:
            v_t = st.selectbox("Variabel Utama:", selected_num_cols)
            v1, v2 = st.columns(2)
            with v1:
                fig_h = px.histogram(df, x=v_t, marginal="rug", title=f"Histogram: {v_t}")
                st.plotly_chart(fig_h, use_container_width=True)
            with v2:
                fig_b = px.box(df, y=v_t, points="all", title=f"Boxplot: {v_t}")
                st.plotly_chart(fig_b, use_container_width=True)

            st.markdown("#### Tabel Distribusi Frekuensi")
            st.dataframe(build_frequency_table(df[v_t]), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("💡 Interpretasi Otomatis")
        if selected_num_cols:
            for col in selected_num_cols:
                s = df[col].dropna()
                if len(s) == 0: continue
                st.markdown(f"**{col}**: Rata-rata = **{s.mean():.{decimal_places}f}**, Median = **{s.median():.{decimal_places}f}**, Std Dev = **{s.std(ddof=ddof_val):.{decimal_places}f}**, Rentang = **{s.min():.{decimal_places}f} - {s.max():.{decimal_places}f}**.")

    with tab5:
        st.subheader("📥 Ekspor Hasil")
        if selected_num_cols:
            st.download_button("Unduh CSV Ringkasan", data=sum_df.to_csv().encode('utf-8'), file_name="ringkasan.csv", mime="text/csv")
else:
    st.info("👋 Silakan unggah file CSV atau Excel di sidebar kiri untuk mulai menggunakan StatistikaLab.")
