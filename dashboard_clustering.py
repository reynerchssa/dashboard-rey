"""
Dashboard Analisis Clustering Kecamatan Surabaya
================================================
Dashboard interaktif untuk analisis segmentasi wilayah menggunakan K-Means Clustering
berdasarkan data kemiskinan, beasiswa, demografi, dan infrastruktur bangunan.

Author: Rey
Tools: Python, Streamlit, Scikit-learn, Pandas, Plotly
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

import streamlit as st          # Framework untuk membuat web app interaktif
import pandas as pd             # Library untuk manipulasi dan analisis data
import numpy as np              # Library untuk operasi numerik dan array
import plotly.express as px     # Library untuk visualisasi interaktif (high-level)
import plotly.graph_objects as go  # Library untuk visualisasi interaktif (low-level)
from sklearn.preprocessing import StandardScaler  # Untuk normalisasi data
from sklearn.cluster import KMeans                # Algoritma K-Means clustering
from sklearn.metrics import silhouette_score      # Metrik evaluasi clustering
import warnings
warnings.filterwarnings('ignore')  # Menyembunyikan warning messages

# ============================================================================
# KONFIGURASI APLIKASI
# ============================================================================

st.set_page_config(
    page_title="Dashboard Analisis Clustering Kecamatan Surabaya",  # Judul tab browser
    layout="wide"  # Layout lebar penuh untuk dashboard
)

# ============================================================================
# FUNGSI LOADING DAN AGREGASI DATA BANGUNAN
# ============================================================================

@st.cache_data
def load_and_aggregate_building_data():
    """
    Load dan agregasi data bangunan per kecamatan
    Handle format numerik: 1 = Ya/Layak, 0 = Tidak
    """
    try:
        # Coba baca file dengan berbagai nama
        file_loaded = None
        try:
            df_bangunan = pd.read_excel('Data-bangunan-Kirim-UPN.xlsx')
            file_loaded = 'Data-bangunan-Kirim-UPN.xlsx'
        except:
            try:
                df_bangunan = pd.read_excel('Data-bangunan.xlsx')
                file_loaded = 'Data-bangunan.xlsx'
            except:
                return None
        
        # ========================================
        # PENTING: Standarisasi nama kecamatan ke UPPERCASE
        # ========================================
        df_bangunan['kecamatan'] = df_bangunan['kecamatan'].str.upper().str.strip()
        
        # ========================================
        # AGREGASI 1: Total Bangunan per Kecamatan
        # ========================================
        df_agg = df_bangunan.groupby('kecamatan').agg({
            'id_bangunan': 'count'
        }).reset_index()
        df_agg.columns = ['kecamatan', 'total_bangunan']
        
        # ========================================
        # AGREGASI 2: Persentase Kelayakan (NUMERIK: 1 atau 0)
        # ========================================
        if 'status_kelayakan' in df_bangunan.columns:
            # PENTING: Konversi ke numeric untuk handle string '1' atau '0'
            df_bangunan['status_kelayakan'] = pd.to_numeric(
                df_bangunan['status_kelayakan'], 
                errors='coerce'  # Ubah error jadi NaN
            ).fillna(0)  # NaN jadi 0
            
            # Hitung persentase yang bernilai 1 (Layak)
            kelayakan = df_bangunan.groupby('kecamatan')['status_kelayakan'].apply(
                lambda x: (x == 1).sum() / len(x) * 100 if len(x) > 0 else 0
            ).reset_index()
            kelayakan.columns = ['kecamatan', 'persen_layak']
            
            # Merge ke agregat
            df_agg = df_agg.merge(kelayakan, on='kecamatan', how='left')
        
        # ========================================
        # AGREGASI 3: Persentase Huni (NUMERIK: 1 atau 0)
        # ========================================
        if 'status_huni' in df_bangunan.columns:
            df_bangunan['status_huni'] = pd.to_numeric(
                df_bangunan['status_huni'], 
                errors='coerce'
            ).fillna(0)
            
            huni = df_bangunan.groupby('kecamatan')['status_huni'].apply(
                lambda x: (x == 1).sum() / len(x) * 100 if len(x) > 0 else 0
            ).reset_index()
            huni.columns = ['kecamatan', 'persen_berpenghuni']
            
            df_agg = df_agg.merge(huni, on='kecamatan', how='left')
        
        # ========================================
        # AGREGASI 4: Persentase Rumah Tinggal (TEXT)
        # ========================================
        if 'peruntukan' in df_bangunan.columns:
            rumah = df_bangunan.groupby('kecamatan')['peruntukan'].apply(
                lambda x: x.str.contains('Rumah|Tinggal', case=False, na=False).sum() / len(x) * 100 if len(x) > 0 else 0
            ).reset_index()
            rumah.columns = ['kecamatan', 'persen_rumah_tinggal']
            
            df_agg = df_agg.merge(rumah, on='kecamatan', how='left')
        
        # ========================================
        # AGREGASI 5: Keragaman Jenis Kegiatan
        # ========================================
        if 'jenis_kegiatan' in df_bangunan.columns:
            keragaman = df_bangunan.groupby('kecamatan')['jenis_kegiatan'].nunique().reset_index()
            keragaman.columns = ['kecamatan', 'keragaman_kegiatan']
            
            df_agg = df_agg.merge(keragaman, on='kecamatan', how='left')
        
        return df_agg
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading bangunan: {str(e)[:100]}")
        return None

# ============================================================================
# FUNGSI LOADING DAN MERGE SEMUA DATASET
# ============================================================================

@st.cache_data  # Caching untuk performa
def load_data():
    """
    Load dan merge semua dataset utama untuk analisis

    Fungsi ini membaca 5 file Excel berbeda dan menggabungkannya menjadi
    satu DataFrame komprehensif dengan menambahkan metrik kalkulasi.

    Proses:
    1. Load data kemiskinan, beasiswa, dan demografi
    2. Standarisasi nama kecamatan di semua dataset
    3. Merge semua dataset berdasarkan kolom 'kecamatan'
    4. Load dan merge data bangunan (jika tersedia)
    5. Hitung metrik tambahan (rasio, gap, density)
    6. Handle missing values

    Returns:
        DataFrame: Data gabungan dengan kolom:
                   Dari data kemiskinan:
                   - jumlah_warga, warga_miskin, persentase_miskin
                   Dari data beasiswa:
                   - penerima_beasiswa, persentase_penerima
                   Dari data demografi:
                   - umur_produktif, umur_anak, umur_lansia, persentase_produktif
                   Dari data bangunan:
                   - total_bangunan, persen_layak, persen_berpenghuni, dll
                   Metrik kalkulasi:
                   - rasio_beasiswa_kemiskinan: persentase_penerima / persentase_miskin
                   - gap_bantuan: persentase_miskin - persentase_penerima
                   - density_bangunan: (total_bangunan / jumlah_warga) * 1000
    """

    # Load data dari 3 file Excel utama
    df_miskin = pd.read_excel('Data-Warga-miskin.xlsx')        # Data kemiskinan per kecamatan
    df_beasiswa = pd.read_excel('Data-penerima-beasiswa.xlsx', sheet_name='Sheet1')  # Data beasiswa
    df_usia = pd.read_excel('Data-warga-by-usia.xlsx')         # Data demografi usia

    # Standarisasi nama kecamatan di semua dataset
    # Ini PENTING untuk memastikan merge berhasil (case-sensitive)
    df_miskin['kecamatan'] = df_miskin['kecamatan'].str.upper().str.strip()
    df_beasiswa['kecamatan'] = df_beasiswa['kecamatan'].str.upper().str.strip()
    df_usia['kecamatan'] = df_usia['kecamatan'].str.upper().str.strip()

    # Gunakan data kemiskinan sebagai base DataFrame
    df_merged = df_miskin.copy()

    # Merge dengan data beasiswa (left join untuk keep semua kecamatan)
    df_merged = df_merged.merge(
        df_beasiswa[['kecamatan', 'penerima_beasiswa', 'persentase_penerima']], 
        on='kecamatan',  # Join key
        how='left'       # Left join: keep all rows from left table
    )

    # Merge dengan data demografi usia
    df_merged = df_merged.merge(
        df_usia[['kecamatan', 'umur_produktif', 'umur_anak', 'umur_lansia', 'persentase_produktif']], 
        on='kecamatan', 
        how='left'
    )

    # Load dan merge data bangunan (jika file tersedia)
    df_bangunan_agg = load_and_aggregate_building_data()
    if df_bangunan_agg is not None:
        df_merged = df_merged.merge(df_bangunan_agg, on='kecamatan', how='left')

        # Hitung density bangunan per 1000 penduduk
        # Ini metrik penting untuk mengukur kepadatan pembangunan
        df_merged['density_bangunan'] = (df_merged['total_bangunan'] / df_merged['jumlah_warga'] * 1000).fillna(0)

    # Hitung rasio beasiswa terhadap kemiskinan
    # Rasio ideal = 1 (beasiswa seimbang dengan kemiskinan)
    # Rasio < 1 = beasiswa kurang, Rasio > 1 = beasiswa lebih dari cukup
    # +0.01 untuk menghindari division by zero
    df_merged['rasio_beasiswa_kemiskinan'] = df_merged['persentase_penerima'] / (df_merged['persentase_miskin'] + 0.01)

    # Hitung gap antara kemiskinan dan penerima beasiswa
    # Gap positif = kekurangan beasiswa, Gap negatif = kelebihan beasiswa
    df_merged['gap_bantuan'] = df_merged['persentase_miskin'] - df_merged['persentase_penerima']

    # Fill missing values dengan 0 untuk menghindari error dalam visualisasi
    # NaN dalam plotting dapat menyebabkan error "Invalid element received"
    df_merged = df_merged.fillna(0)

    return df_merged

# ============================================================================
# FUNGSI K-MEANS CLUSTERING
# ============================================================================

def perform_clustering(df, n_clusters, features):
    """
    Melakukan K-Means clustering pada data dengan fitur yang dipilih

    K-Means adalah algoritma unsupervised learning yang mengelompokkan data
    menjadi k cluster berdasarkan kesamaan (similarity) dalam feature space.

    Proses:
    1. Filter fitur yang valid dan tersedia
    2. Extract data untuk clustering
    3. Normalisasi menggunakan StandardScaler (penting untuk K-Means)
    4. Fit model K-Means
    5. Prediksi cluster untuk setiap kecamatan
    6. Evaluasi kualitas clustering dengan Silhouette Score

    Args:
        df (DataFrame): Data input dengan semua fitur
        n_clusters (int): Jumlah cluster yang diinginkan (k)
        features (list): List nama kolom yang digunakan untuk clustering

    Returns:
        tuple: (df_with_cluster, kmeans_model, scaler, silhouette_score)
               - df_with_cluster: DataFrame dengan kolom 'cluster' baru (0, 1, 2, ...)
               - kmeans_model: Model KMeans yang sudah difit (untuk analisis lebih lanjut)
               - scaler: StandardScaler yang digunakan (untuk transformasi data baru)
               - silhouette_score: Skor kualitas clustering (0-1, higher is better)
    """

    # Copy dataframe untuk menghindari modifikasi data asli
    df_cluster = df.copy()

    # Filter hanya fitur yang tersedia di dataframe
    # Ini penting karena beberapa fitur (seperti data bangunan) mungkin tidak tersedia
    available_features = [f for f in features if f in df_cluster.columns]

    # Validasi: minimal 2 fitur diperlukan untuk clustering
    if len(available_features) < 2:
        st.error("Tidak cukup fitur valid untuk clustering")
        return df_cluster, None, None, 0

    # Extract data untuk clustering (convert ke numpy array)
    X = df_cluster[available_features].values

    # Normalisasi data menggunakan StandardScaler
    # StandardScaler: mentransformasi data menjadi mean=0 dan std=1
    # Ini PENTING untuk K-Means karena algoritma sensitive terhadap skala data
    # Contoh: "jumlah_warga" (puluhan ribu) vs "persentase_miskin" (0-10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Inisialisasi dan fit model K-Means
    # random_state=42: untuk reproducibility (hasil konsisten)
    # n_init=10: jumlah inisialisasi dengan centroid berbeda
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Prediksi cluster untuk setiap data point
    # Hasil: array dengan nilai 0, 1, 2, ..., (n_clusters-1)
    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

    # Hitung Silhouette Score untuk evaluasi kualitas clustering
    # Score range: -1 to 1
    # Score mendekati 1: cluster well-separated dan compact
    # Score mendekati 0: cluster overlapping
    # Score negatif: data mungkin di cluster yang salah
    silhouette = silhouette_score(X_scaled, df_cluster['cluster'])

    return df_cluster, kmeans, scaler, silhouette

# ============================================================================
# FUNGSI KLASIFIKASI WILAYAH BERDASARKAN KEMISKINAN
# ============================================================================

def classify_region(row):
    """
    Klasifikasi wilayah berdasarkan tingkat kemiskinan

    Threshold ditentukan berdasarkan distribusi data kemiskinan di Surabaya:
    - < 1%: Wilayah relatif sejahtera
    - 1-2.5%: Wilayah menengah dengan kemiskinan sedang
    - > 2.5%: Wilayah dengan kemiskinan relatif tinggi

    Args:
        row (Series): Baris data dengan kolom 'persentase_miskin'

    Returns:
        str: Klasifikasi wilayah (Menengah Atas/Menengah/Menengah Bawah)
    """
    if row['persentase_miskin'] < 1.0:
        return 'Menengah Atas'
    elif row['persentase_miskin'] < 2.5:
        return 'Menengah'
    else:
        return 'Menengah Bawah'

# ============================================================================
# FUNGSI ANALISIS KESESUAIAN SASARAN BEASISWA
# ============================================================================

def analyze_target_accuracy(row):
    """
    Analisis kesesuaian sasaran beasiswa dengan tingkat kemiskinan

    Fungsi ini mengevaluasi apakah distribusi beasiswa sudah sesuai dengan
    tingkat kemiskinan di wilayah tersebut. Beberapa skenario:

    1. Tidak Sesuai - Perlu Prioritas:
       Kemiskinan tinggi (>3%) tapi beasiswa rendah (<0.3%)
       Wilayah ini membutuhkan perhatian khusus dan peningkatan beasiswa

    2. Tidak Sesuai - Over Alokasi:
       Kemiskinan rendah (<1%) tapi beasiswa tinggi (>0.4%)
       Wilayah ini menerima beasiswa berlebih, perlu evaluasi

    3. Kurang Sesuai - Perlu Ditingkatkan:
       Rasio beasiswa/kemiskinan terlalu rendah (<0.3)
       Beasiswa hanya mencakup <30% dari tingkat kemiskinan

    4. Baik:
       Rasio beasiswa/kemiskinan tinggi (>1.5)
       Beasiswa sudah melebihi tingkat kemiskinan

    5. Cukup Sesuai:
       Rasio sedang (0.3-1.5), distribusi cukup seimbang

    Args:
        row (Series): Baris data dengan kolom:
                      - persentase_miskin
                      - persentase_penerima
                      - rasio_beasiswa_kemiskinan

    Returns:
        str: Status kesesuaian sasaran beasiswa
    """

    # Skenario 1: Kemiskinan tinggi tapi beasiswa rendah (PRIORITAS)
    if row['persentase_miskin'] > 3.0 and row['persentase_penerima'] < 0.3:
        return 'Tidak Sesuai - Perlu Prioritas'

    # Skenario 2: Kemiskinan rendah tapi beasiswa tinggi (OVER ALOKASI)
    elif row['persentase_miskin'] < 1.0 and row['persentase_penerima'] > 0.4:
        return 'Tidak Sesuai - Over Alokasi'

    # Skenario 3: Rasio terlalu rendah (PERLU DITINGKATKAN)
    elif row['rasio_beasiswa_kemiskinan'] < 0.3:
        return 'Kurang Sesuai - Perlu Ditingkatkan'

    # Skenario 4: Rasio bagus (BAIK)
    elif row['rasio_beasiswa_kemiskinan'] > 1.5:
        return 'Baik'

    # Skenario 5: Rasio sedang (CUKUP SESUAI)
    else:
        return 'Cukup Sesuai'

# ============================================================================
# FUNGSI KLASIFIKASI KONDISI INFRASTRUKTUR
# ============================================================================

def classify_infrastructure(row):
    """
    Klasifikasi kondisi infrastruktur berdasarkan kelayakan bangunan

    Threshold berdasarkan standar kelayakan hunian:
    - > 80%: Mayoritas bangunan layak (kondisi baik)
    - 60-80%: Sebagian bangunan layak (kondisi sedang)
    - < 60%: Banyak bangunan tidak layak (perlu perbaikan)

    Args:
        row (Series): Baris data dengan kolom 'persen_layak'

    Returns:
        str: Klasifikasi kondisi infrastruktur
    """

    # Check apakah data tersedia dan valid
    if 'persen_layak' not in row or row['persen_layak'] == 0:
        return 'Data Tidak Tersedia'

    # Klasifikasi berdasarkan threshold
    if row['persen_layak'] > 80:
        return 'Infrastruktur Baik'
    elif row['persen_layak'] > 60:
        return 'Infrastruktur Sedang'
    else:
        return 'Infrastruktur Perlu Perbaikan'

# ============================================================================
# HEADER APLIKASI
# ============================================================================

# Judul utama dashboard
st.title("Dashboard Analisis Clustering Kecamatan Surabaya")

# Subjudul dengan deskripsi singkat
st.markdown("Analisis distribusi beasiswa, kemiskinan, demografi, dan kondisi infrastruktur bangunan")

# Garis pembatas
st.markdown("---")

# ============================================================================
# LOAD DATA DAN PREPROCESSING
# ============================================================================

# Tampilkan spinner saat loading data untuk UX yang lebih baik
with st.spinner("Loading dan processing data..."):
    df = load_data()

# Terapkan fungsi klasifikasi pada setiap baris data
# apply() menjalankan fungsi pada setiap row
df['klasifikasi_wilayah'] = df.apply(classify_region, axis=1)
df['status_sasaran'] = df.apply(analyze_target_accuracy, axis=1)
df['kondisi_infrastruktur'] = df.apply(classify_infrastructure, axis=1)

# ============================================================================
# SIDEBAR - PARAMETER CLUSTERING
# ============================================================================

# Header sidebar
st.sidebar.header("Pengaturan Clustering")
st.sidebar.markdown("### Parameter K-Means")

# Slider untuk memilih jumlah cluster (k)
# min_value=2: minimal 2 cluster
# max_value=6: maksimal 6 cluster (lebih dari 6 biasanya terlalu banyak untuk 31 kecamatan)
# value=4: default value
n_clusters = st.sidebar.slider("Jumlah Cluster (k)", min_value=2, max_value=6, value=4, step=1)

st.sidebar.markdown("### Fitur untuk Clustering")

# Dictionary mapping nama fitur (user-friendly) ke nama kolom (technical)
available_features = {
    'Persentase Miskin': 'persentase_miskin',
    'Persentase Penerima Beasiswa': 'persentase_penerima',
    'Gap Bantuan': 'gap_bantuan',
    'Rasio Beasiswa/Kemiskinan': 'rasio_beasiswa_kemiskinan',
    'Persentase Produktif': 'persentase_produktif',
    'Jumlah Warga': 'jumlah_warga'
}

# Tambahkan fitur infrastruktur jika data tersedia
# Check apakah kolom ada DAN memiliki nilai (tidak semua 0)
if 'total_bangunan' in df.columns and df['total_bangunan'].sum() > 0:
    available_features.update({
        'Total Bangunan': 'total_bangunan',
        'Density Bangunan': 'density_bangunan',
    })
    # Tambahkan persen_layak jika tersedia
    if 'persen_layak' in df.columns:
        available_features['Persentase Bangunan Layak'] = 'persen_layak'

# Multiselect untuk memilih fitur yang akan digunakan dalam clustering
# User dapat memilih 2 atau lebih fitur
selected_feature_names = st.sidebar.multiselect(
    "Pilih Fitur (minimal 2)",
    list(available_features.keys()),  # Options (nama user-friendly)
    default=['Persentase Miskin', 'Persentase Penerima Beasiswa', 'Persentase Produktif']  # Default selection
)

# Validasi: minimal 2 fitur diperlukan untuk clustering
if len(selected_feature_names) < 2:
    st.warning("Pilih minimal 2 fitur untuk melakukan clustering")
    st.stop()  # Hentikan eksekusi jika validasi gagal

# Convert nama fitur user-friendly ke nama kolom technical
selected_features = [available_features[name] for name in selected_feature_names]

# ============================================================================
# EKSEKUSI CLUSTERING
# ============================================================================

# Jalankan fungsi clustering dengan parameter yang dipilih user
df_clustered, kmeans, scaler, silhouette = perform_clustering(df, n_clusters, selected_features)

# ============================================================================
# KEY PERFORMANCE INDICATORS (KPI) - ROW PERTAMA
# ============================================================================

st.subheader("Key Performance Indicators")

# Buat 4 kolom dengan lebar sama untuk menampilkan 4 KPI
col1, col2, col3, col4 = st.columns(4)

# KPI 1: Total Kecamatan
with col1:
    st.metric(
        "Total Kecamatan", 
        len(df_clustered),  # Jumlah total kecamatan
        help="Jumlah kecamatan di Surabaya"  # Tooltip saat hover
    )

# KPI 2: Kesesuaian Sasaran
with col2:
    # Hitung jumlah kecamatan dengan status "Baik" atau "Cukup Sesuai"
    sesuai = len(df_clustered[df_clustered['status_sasaran'].isin(['Baik', 'Cukup Sesuai'])])
    # Hitung persentase
    persen_sesuai = sesuai / len(df_clustered) * 100
    st.metric(
        "Kesesuaian Sasaran", 
        f"{sesuai}/31",  # Format: X/31
        f"{persen_sesuai:.1f}%"  # Delta menunjukkan persentase
    )

# KPI 3: Wilayah Prioritas
with col3:
    # Hitung jumlah wilayah yang perlu prioritas
    prioritas = len(df_clustered[df_clustered['status_sasaran'] == 'Tidak Sesuai - Perlu Prioritas'])
    st.metric(
        "Wilayah Prioritas", 
        prioritas, 
        help="Wilayah yang perlu perhatian khusus"
    )

# KPI 4: Silhouette Score (Kualitas Clustering)
with col4:
    st.metric(
        "Silhouette Score", 
        f"{silhouette:.3f}",  # Format 3 desimal
        help="Kualitas clustering (0-1, semakin tinggi semakin baik)"
    )

# Garis pembatas setelah KPI
st.markdown("---")

# ============================================================================
# TABS NAVIGASI - 6 TAB UTAMA
# ============================================================================

# Buat 6 tabs untuk organisasi konten yang lebih baik
tabs = st.tabs([
    "Overview & Clustering",      # Tab 1: Hasil clustering dan karakteristik cluster
    "Analisis Sasaran Beasiswa",  # Tab 2: Analisis kesesuaian beasiswa vs kemiskinan
    "Analisis Infrastruktur",     # Tab 3: Analisis kondisi bangunan
    "Visualisasi Interaktif",     # Tab 4: Scatter plot dan heatmap custom
    "Data & Export",              # Tab 5: Tabel data dan download
    "Insights & Rekomendasi"      # Tab 6: Insight dan rekomendasi kebijakan
])

# ============================================================================
# TAB 1: OVERVIEW & CLUSTERING
# ============================================================================

with tabs[0]:
    st.subheader("Hasil Clustering K-Means")

    # Buat 2 kolom: kiri lebih lebar (2:1 ratio)
    col1, col2 = st.columns([2, 1])

    # KOLOM KIRI: Bar Chart Distribusi Cluster
    with col1:
        st.markdown("### Distribusi Cluster")

        # Hitung jumlah kecamatan per cluster
        cluster_counts = df_clustered['cluster'].value_counts().sort_index()

        # Buat bar chart menggunakan Plotly Graph Objects
        fig_cluster = go.Figure()
        fig_cluster.add_trace(go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],  # Label sumbu X
            y=cluster_counts.values,  # Nilai sumbu Y (jumlah kecamatan)
            # Warna berbeda untuk setiap cluster (palet Plotly default)
            marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'][:len(cluster_counts)],
            text=cluster_counts.values,  # Label pada bar
            textposition='outside'  # Posisi label di luar bar
        ))
        fig_cluster.update_layout(
            title='Jumlah Kecamatan per Cluster',
            xaxis_title='Cluster',
            yaxis_title='Jumlah Kecamatan',
            height=400  # Tinggi chart dalam pixels
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    # KOLOM KANAN: Pie Chart Proporsi Cluster
    with col2:
        st.markdown("### Proporsi Cluster")

        # Buat donut chart (pie chart dengan hole)
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            hole=0.4  # Hole size untuk donut chart (0 = pie chart penuh)
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Garis pembatas
    st.markdown("---")
    st.subheader("Karakteristik Setiap Cluster")

    # Loop untuk setiap cluster dan tampilkan karakteristiknya
    for cluster in sorted(df_clustered['cluster'].unique()):
        # Filter data untuk cluster ini
        cluster_data = df_clustered[df_clustered['cluster']==cluster]

        # Buat expander (collapsible section) untuk setiap cluster
        # expanded=(cluster==0): hanya cluster 0 yang terbuka secara default
        with st.expander(f"Cluster {cluster} - {len(cluster_data)} Kecamatan", expanded=(cluster==0)):

            # Buat 4 kolom untuk menampilkan 4 metrics
            cols = st.columns(4)

            # Metric 1: Rata-rata Persentase Miskin
            cols[0].metric(
                "Rata-rata Persentase Miskin", 
                f"{cluster_data['persentase_miskin'].mean():.2f}%"
            )

            # Metric 2: Rata-rata Persentase Beasiswa
            cols[1].metric(
                "Rata-rata Persentase Beasiswa", 
                f"{cluster_data['persentase_penerima'].mean():.2f}%"
            )

            # Metric 3: Rata-rata Gap
            cols[2].metric(
                "Rata-rata Gap", 
                f"{cluster_data['gap_bantuan'].mean():.2f}%"
            )

            # Metric 4: Rata-rata Persentase Produktif
            cols[3].metric(
                "Rata-rata Persentase Produktif", 
                f"{cluster_data['persentase_produktif'].mean():.1f}%"
            )

            # Cari nilai dominan (mode) untuk klasifikasi dan status
            # mode()[0] mengambil nilai yang paling sering muncul
            dom_klasifikasi = cluster_data['klasifikasi_wilayah'].mode()[0]
            dom_status = cluster_data['status_sasaran'].mode()[0]

            # Tampilkan klasifikasi dan status dominan dalam 2 kolom
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Klasifikasi Ekonomi Dominan:** {dom_klasifikasi}")
            with col2:
                st.info(f"**Status Sasaran Dominan:** {dom_status}")

            # Tampilkan daftar kecamatan dalam cluster
            st.markdown("**Daftar Kecamatan:**")
            kecamatan_list = sorted(cluster_data['kecamatan'].values)  # Sort alphabetically
            st.write(", ".join(kecamatan_list))  # Join dengan koma

# ============================================================================
# TAB 2: ANALISIS SASARAN BEASISWA
# ============================================================================

with tabs[1]:
    st.subheader("Analisis Kesesuaian Sasaran Beasiswa")

    # Buat 2 kolom untuk 2 visualisasi
    col1, col2 = st.columns(2)

    # KOLOM KIRI: Scatter Plot Kemiskinan vs Beasiswa
    with col1:
        # Filter data yang valid untuk plotting (jumlah_warga > 0)
        # Ini untuk menghindari error NaN atau Inf dalam visualisasi
        plot_data = df_clustered[df_clustered['jumlah_warga'] > 0].copy()

        # Buat scatter plot dengan Plotly Express
        fig_scatter = px.scatter(
            plot_data,
            x='persentase_miskin',      # Sumbu X: tingkat kemiskinan
            y='persentase_penerima',    # Sumbu Y: penerima beasiswa
            size='jumlah_warga',        # Ukuran bubble berdasarkan populasi
            color='status_sasaran',     # Warna berdasarkan status sasaran
            hover_name='kecamatan',     # Label saat hover
            labels={
                'persentase_miskin': 'Persentase Warga Miskin (%)',
                'persentase_penerima': 'Persentase Penerima Beasiswa (%)'
            },
            title='Kemiskinan vs Penerima Beasiswa',
            # Custom color mapping untuk setiap status
            color_discrete_map={
                'Baik': '#00CC96',                                # Hijau
                'Cukup Sesuai': '#FFA15A',                       # Orange
                'Kurang Sesuai - Perlu Ditingkatkan': '#EF553B', # Merah
                'Tidak Sesuai - Perlu Prioritas': '#AB63FA',     # Ungu
                'Tidak Sesuai - Over Alokasi': '#FF6692'         # Pink
            }
        )

        # Tambahkan garis ideal (diagonal) untuk referensi
        # Garis ini menunjukkan kondisi ideal: beasiswa = kemiskinan (rasio 1:1)
        max_val = max(plot_data['persentase_miskin'].max(), plot_data['persentase_penerima'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val],  # Dari (0,0) ke (max, max)
            y=[0, max_val],
            mode='lines',
            name='Garis Ideal (1:1)',
            line=dict(dash='dash', color='gray', width=2),  # Garis putus-putus abu-abu
            showlegend=True
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # KOLOM KANAN: Pie Chart Status Sasaran
    with col2:
        # Hitung distribusi status sasaran
        status_counts = df_clustered['status_sasaran'].value_counts()

        # Buat donut chart
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Distribusi Status Kesesuaian Sasaran',
            hole=0.3  # Donut chart
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # Garis pembatas
    st.markdown("---")
    st.subheader("Rasio Beasiswa terhadap Kemiskinan")

    # Sort data berdasarkan rasio (descending)
    df_sorted = df_clustered.sort_values('rasio_beasiswa_kemiskinan', ascending=False)

    # Tentukan warna bar berdasarkan nilai rasio
    # Hijau (>0.5), Orange (0.3-0.5), Merah (<0.3)
    colors = ['green' if x > 0.5 else 'orange' if x > 0.3 else 'red' 
              for x in df_sorted['rasio_beasiswa_kemiskinan']]

    # Buat bar chart horizontal
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_sorted['kecamatan'],
        y=df_sorted['rasio_beasiswa_kemiskinan'],
        marker_color=colors,  # Warna custom per bar
        text=df_sorted['rasio_beasiswa_kemiskinan'].round(2),  # Label value
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Rasio: %{y:.2f}<extra></extra>'  # Custom hover text
    ))

    # Tambahkan garis horizontal untuk target ideal (rasio 0.5)
    fig_bar.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="Target Ideal (0.5)",  # Label garis
        annotation_position="right"
    )

    fig_bar.update_layout(
        title='Rasio Penerima Beasiswa / Kemiskinan per Kecamatan',
        xaxis_title='Kecamatan',
        yaxis_title='Rasio',
        xaxis_tickangle=-45,  # Rotate label 45 derajat untuk readability
        height=500,
        showlegend=False  # Tidak perlu legend untuk single trace
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Info box untuk interpretasi rasio
    st.info("""
    **Interpretasi Rasio:**
    - Hijau (> 0.5): Distribusi beasiswa baik (beasiswa mencakup > 50% dari tingkat kemiskinan)
    - Orange (0.3 - 0.5): Cukup, tapi masih perlu ditingkatkan
    - Merah (< 0.3): Kurang sesuai sasaran, perlu menjadi prioritas
    """)

# ============================================================================
# TAB 3: ANALISIS INFRASTRUKTUR BANGUNAN
# ============================================================================

with tabs[2]:
    st.subheader("🏢 Analisis Kondisi Infrastruktur Bangunan")
    
    # ========================================
    # DEBUG INFO
    # ========================================
    with st.expander("🔍 Debug Info - Cek Status Data Bangunan", expanded=False):
        st.write("**Status Kolom Data Bangunan:**")
        
        if 'total_bangunan' in df_clustered.columns:
            st.success("✅ Kolom 'total_bangunan' ADA")
            st.write(f"- Total bangunan: {df_clustered['total_bangunan'].sum():,.0f}")
            st.write(f"- Max per kecamatan: {df_clustered['total_bangunan'].max():,.0f}")
            st.write(f"- Kecamatan dengan data: {(df_clustered['total_bangunan'] > 0).sum()}/31")
        else:
            st.error("❌ Kolom 'total_bangunan' TIDAK ADA")
        
        if 'persen_layak' in df_clustered.columns:
            st.success("✅ Kolom 'persen_layak' ADA")
            st.write(f"- Rata-rata kelayakan: {df_clustered['persen_layak'].mean():.2f}%")
            st.write(f"- Min: {df_clustered['persen_layak'].min():.2f}%")
            st.write(f"- Max: {df_clustered['persen_layak'].max():.2f}%")
            st.write(f"- Kecamatan dengan data > 0: {(df_clustered['persen_layak'] > 0).sum()}/31")
            
            # Tambahan: Cek distribusi nilai
            st.write(f"- Jumlah nilai 0%: {(df_clustered['persen_layak'] == 0).sum()}")
            st.write(f"- Jumlah nilai 100%: {(df_clustered['persen_layak'] == 100).sum()}")
            st.write(f"- Jumlah nilai di antaranya: {((df_clustered['persen_layak'] > 0) & (df_clustered['persen_layak'] < 100)).sum()}")
        else:
            st.error("❌ Kolom 'persen_layak' TIDAK ADA")
        
        st.write("\n**Sample Data (5 kecamatan):**")
        cols_to_show = ['kecamatan', 'total_bangunan']
        if 'persen_layak' in df_clustered.columns:
            cols_to_show.append('persen_layak')
        if 'density_bangunan' in df_clustered.columns:
            cols_to_show.append('density_bangunan')
        st.dataframe(df_clustered[cols_to_show].head())
    
    # ========================================
    # CEK KETERSEDIAAN DATA
    # ========================================
    has_building_data = (
        'total_bangunan' in df_clustered.columns and 
        df_clustered['total_bangunan'].sum() > 0
    )
    
    if not has_building_data:
        # ========================================
        # TAMPILAN JIKA DATA TIDAK TERSEDIA
        # ========================================
        st.warning("⚠️ Data bangunan tidak tersedia atau gagal dimuat.")
        st.info("📁 Pastikan file 'Data-bangunan.xlsx' atau 'Data-bangunan-Kirim-UPN.xlsx' ada di folder yang sama dengan dashboard ini.")
        
        with st.expander("📖 Panduan Troubleshooting"):
            st.markdown("""
            ### Penyebab Umum:
            1. **File tidak ditemukan** di folder project
            2. **File terlalu besar** (>100MB) menyebabkan timeout
            3. **Kolom tidak sesuai** - Missing: `kecamatan`, `id_bangunan`, `status_kelayakan`
            4. **Nama kecamatan tidak match** - Harus UPPERCASE dan konsisten
            
            ### Cara Memperbaiki:
            
            #### 1. Cek file:
            ```
            dir *.xlsx | findstr bangunan
            ```
            
            #### 2. Kolom wajib:
            - `kecamatan` → "ASEM ROWO" (UPPERCASE)
            - `id_bangunan` → ID unik
            - `status_kelayakan` → **1** atau **0** (NUMERIC!)
            
            #### 3. Format status_kelayakan:
            - ✅ Benar: 1, 0 (numerik)
            - ❌ Salah: "1", "0" (string), "Layak", "Ya"
            
            #### 4. Cek nama kecamatan match:
            Nama di file bangunan HARUS sama dengan file lain:
            - Data-Warga-miskin.xlsx
            - Data-penerima-beasiswa.xlsx
            
            Semua harus UPPERCASE: "ASEM ROWO", bukan "Asem Rowo"
            """)
        
        st.info("💡 **Dashboard tetap berfungsi** untuk 5 tab lainnya.")
        
    else:
        # ========================================
        # TAMPILAN JIKA DATA TERSEDIA
        # ========================================
        
        # KPI Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_bg = df_clustered['total_bangunan'].sum()
            st.metric(
                label="📊 Total Bangunan",
                value=f"{int(total_bg):,}",
                help="Total seluruh bangunan di 31 kecamatan Surabaya"
            )
        
        with col2:
            if 'persen_layak' in df_clustered.columns:
                avg_layak = df_clustered['persen_layak'].mean()
                
                if avg_layak == 0:
                    st.metric(label="✅ Rata-rata Kelayakan", value="0.0%")
                    st.error("⚠️ SEMUA bangunan tidak layak ATAU data encoding salah!")
                    st.caption("Cek: apakah 1=Layak sudah benar di raw data?")
                else:
                    delta_text = "Sangat Baik" if avg_layak >= 80 else "Baik" if avg_layak >= 60 else "Perlu Perbaikan"
                    st.metric(
                        label="✅ Rata-rata Kelayakan",
                        value=f"{avg_layak:.1f}%",
                        delta=delta_text,
                        help="Persentase bangunan layak huni (status_kelayakan = 1)"
                    )
            else:
                st.metric(label="✅ Rata-rata Kelayakan", value="N/A")
                st.caption("⚠️ Kolom 'status_kelayakan' tidak ada")
        
        with col3:
            if 'density_bangunan' in df_clustered.columns:
                avg_density = df_clustered['density_bangunan'].mean()
                st.metric(
                    label="🏘️ Density Rata-rata",
                    value=f"{avg_density:.1f}",
                    help="Bangunan per 1000 penduduk"
                )
                st.caption("per 1000 penduduk")
            else:
                st.metric(label="🏘️ Density Rata-rata", value="N/A")
        
        st.markdown("---")
        
        # ========================================
        # VISUALISASI ROW 1
        # ========================================
        col1, col2 = st.columns(2)
        
        with col1:
            df_bg_sorted = df_clustered.sort_values('total_bangunan', ascending=False).head(15)
            
            fig_bg = px.bar(
                df_bg_sorted,
                x='kecamatan',
                y='total_bangunan',
                title='🏗️ Top 15 Kecamatan - Jumlah Bangunan',
                color='total_bangunan',
                color_continuous_scale='Blues',
                text='total_bangunan'
            )
            
            fig_bg.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_bg.update_layout(
                xaxis_tickangle=-45,
                height=450,
                showlegend=False,
                xaxis_title="Kecamatan",
                yaxis_title="Jumlah Bangunan"
            )
            
            st.plotly_chart(fig_bg, use_container_width=True)
        
        with col2:
            if 'kondisi_infrastruktur' in df_clustered.columns:
                kondisi_counts = df_clustered['kondisi_infrastruktur'].value_counts()
                
                if len(kondisi_counts) == 1 and kondisi_counts.index[0] == 'Data Tidak Tersedia':
                    st.info("📊 Data kelayakan tidak tersedia")
                    st.caption("Kolom 'kondisi_infrastruktur' dibuat dari 'persen_layak'")
                else:
                    fig_kondisi = px.pie(
                        values=kondisi_counts.values,
                        names=kondisi_counts.index,
                        title='🏘️ Distribusi Kondisi Infrastruktur',
                        hole=0.4,
                        color_discrete_map={
                            'Infrastruktur Baik': '#2ecc71',
                            'Infrastruktur Sedang': '#f39c12',
                            'Infrastruktur Perlu Perbaikan': '#e74c3c',
                            'Data Tidak Tersedia': '#95a5a6'
                        }
                    )
                    
                    fig_kondisi.update_traces(textposition='inside', textinfo='percent+label')
                    fig_kondisi.update_layout(height=450)
                    st.plotly_chart(fig_kondisi, use_container_width=True)
                    
                    baik_pct = (kondisi_counts.get('Infrastruktur Baik', 0) / 31 * 100)
                    if baik_pct >= 60:
                        st.success(f"✅ {baik_pct:.0f}% kecamatan infrastruktur baik")
                    elif baik_pct >= 40:
                        st.warning(f"⚠️ {baik_pct:.0f}% kecamatan infrastruktur baik")
                    else:
                        st.error(f"❌ Hanya {baik_pct:.0f}% kecamatan infrastruktur baik")
            else:
                st.info("📊 Kolom 'kondisi_infrastruktur' tidak tersedia")
        
        # ========================================
        # VISUALISASI ROW 2: Korelasi
        # ========================================
        if 'persen_layak' in df_clustered.columns and df_clustered['persen_layak'].sum() > 0:
            st.markdown("---")
            st.subheader("📈 Korelasi: Kemiskinan vs Kelayakan Bangunan")
            
            plot_data_corr = df_clustered[
                (df_clustered['persen_layak'] > 0) & 
                (df_clustered['total_bangunan'] > 0)
            ].copy()
            
            if len(plot_data_corr) > 0:
                plot_data_corr['size_normalized'] = (
                    plot_data_corr['total_bangunan'] / plot_data_corr['total_bangunan'].max() * 50 + 10
                )
                
                fig_corr = px.scatter(
                    plot_data_corr,
                    x='persentase_miskin',
                    y='persen_layak',
                    size='size_normalized',
                    color='klasifikasi_wilayah',
                    hover_name='kecamatan',
                    hover_data={
                        'total_bangunan': ':,.0f',
                        'size_normalized': False,
                        'persentase_miskin': ':.2f',
                        'persen_layak': ':.2f'
                    },
                    labels={
                        'persentase_miskin': 'Persentase Kemiskinan (%)',
                        'persen_layak': 'Persentase Bangunan Layak (%)',
                        'klasifikasi_wilayah': 'Klasifikasi Wilayah'
                    },
                    title='Hubungan Tingkat Kemiskinan dengan Kelayakan Infrastruktur',
                    color_discrete_map={
                        'Menengah Atas': '#2ecc71',
                        'Menengah': '#f39c12',
                        'Menengah Bawah': '#e74c3c'
                    }
                )
                
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Korelasi Pearson
                correlation = df_clustered[['persentase_miskin', 'persen_layak']].corr().iloc[0, 1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Koefisien Korelasi (r)",
                        value=f"{correlation:.3f}",
                        help="Pearson correlation (-1 hingga +1)"
                    )
                
                with col2:
                    if abs(correlation) >= 0.7:
                        kekuatan = "Sangat Kuat"
                    elif abs(correlation) >= 0.5:
                        kekuatan = "Kuat"
                    elif abs(correlation) >= 0.3:
                        kekuatan = "Sedang"
                    else:
                        kekuatan = "Lemah"
                    st.metric(label="Kekuatan Korelasi", value=kekuatan)
                
                with col3:
                    arah = "Negatif ↘️" if correlation < 0 else "Positif ↗️"
                    st.metric(label="Arah Korelasi", value=arah)
                
                # Interpretasi
                st.markdown("### 📊 Interpretasi:")
                
                if correlation < -0.5:
                    st.error(f"""
                    **Korelasi Negatif Kuat** (r = {correlation:.2f})
                    
                    ⚠️ Wilayah miskin **PASTI** punya bangunan kurang layak.
                    
                    💡 **Rekomendasi:**
                    - Program renovasi massal di wilayah prioritas
                    - Subsidi material bangunan
                    """)
                elif correlation < -0.3:
                    st.warning(f"""
                    **Korelasi Negatif Sedang** (r = {correlation:.2f})
                    
                    ℹ️ Ada kecenderungan wilayah miskin = bangunan kurang layak.
                    
                    💡 **Rekomendasi:**
                    - Review case by case per kecamatan
                    - Program perbaikan bertahap
                    """)
                elif correlation < 0:
                    st.info(f"""
                    **Korelasi Negatif Lemah** (r = {correlation:.2f})
                    
                    ℹ️ Tidak ada pola jelas kemiskinan vs kelayakan.
                    
                    💡 Faktor lain lebih dominan (usia bangunan, zonasi).
                    """)
                else:
                    st.success(f"""
                    **Korelasi Positif** (r = {correlation:.2f})
                    
                    ✅ Wilayah miskin justru punya bangunan layak.
                    
                    Kemungkinan: Program renovasi pemerintah efektif!
                    """)
                
                st.caption(f"📌 Analisis {len(plot_data_corr)} dari 31 kecamatan")
                
            else:
                st.warning("⚠️ Tidak ada data valid untuk korelasi")
        else:
            st.markdown("---")
            st.info("📊 Data kelayakan tidak tersedia untuk analisis korelasi")



# ============================================================================
# TAB 4: VISUALISASI INTERAKTIF
# ============================================================================

with tabs[3]:
    st.subheader("Visualisasi Interaktif Multi-Dimensi")

    st.markdown("### Scatter Plot Custom")
    st.caption("Pilih sumbu X dan Y untuk eksplorasi data")

    # Buat 3 kolom untuk 3 dropdown selector
    col1, col2, col3 = st.columns(3)

    # Dropdown 1: Pilih fitur untuk sumbu X
    with col1:
        x_feature = st.selectbox("Sumbu X", selected_feature_names, index=0, key='viz_x')

    # Dropdown 2: Pilih fitur untuk sumbu Y
    with col2:
        y_feature = st.selectbox(
            "Sumbu Y", 
            selected_feature_names, 
            index=min(1, len(selected_feature_names)-1),  # Default ke fitur kedua
            key='viz_y'
        )

    # Dropdown 3: Pilih variabel untuk pewarnaan
    with col3:
        color_by = st.selectbox(
            "Warna berdasarkan", 
            ['cluster', 'klasifikasi_wilayah', 'status_sasaran'],
            format_func=lambda x: {  # Custom label untuk dropdown
                'cluster': 'Cluster',
                'klasifikasi_wilayah': 'Klasifikasi Wilayah',
                'status_sasaran': 'Status Sasaran'
            }[x]
        )

    # Convert nama fitur ke nama kolom
    x_col = available_features[x_feature]
    y_col = available_features[y_feature]

    # Filter data valid dan normalize size
    plot_data_viz = df_clustered[df_clustered['jumlah_warga'] > 0].copy()
    plot_data_viz['size_viz'] = plot_data_viz['jumlah_warga'] / plot_data_viz['jumlah_warga'].max() * 50 + 10

    # Buat scatter plot custom
    fig_scatter_custom = px.scatter(
        plot_data_viz,
        x=x_col,
        y=y_col,
        color=color_by,  # Pewarnaan berdasarkan pilihan user
        size='size_viz',  # Ukuran bubble
        hover_name='kecamatan',
        hover_data={
            'klasifikasi_wilayah': True,
            'status_sasaran': True,
            'jumlah_warga': ':,',  # Format dengan koma
            'size_viz': False  # Hide normalized size
        },
        labels={x_col: x_feature, y_col: y_feature},
        title=f'{x_feature} vs {y_feature}'
    )
    fig_scatter_custom.update_layout(height=500)
    st.plotly_chart(fig_scatter_custom, use_container_width=True)

    # Garis pembatas
    st.markdown("---")
    st.subheader("Heatmap Korelasi Antar Variabel")

    # Filter fitur yang tersedia
    corr_features = [f for f in selected_features if f in df_clustered.columns]

    if len(corr_features) >= 2:
        # Hitung correlation matrix
        corr_matrix = df_clustered[corr_features].corr()

        # Buat heatmap korelasi
        fig_heatmap = px.imshow(
            corr_matrix,
            labels=dict(x="Variabel", y="Variabel", color="Korelasi"),
            # Label sumbu dengan nama user-friendly
            x=[selected_feature_names[selected_features.index(f)] for f in corr_features],
            y=[selected_feature_names[selected_features.index(f)] for f in corr_features],
            color_continuous_scale='RdBu_r',  # Red-Blue diverging (merah=negatif, biru=positif)
            aspect='auto',  # Auto aspect ratio
            title='Matriks Korelasi Variabel Clustering'
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================================
# TAB 5: DATA & EXPORT
# ============================================================================

with tabs[4]:
    st.subheader("Data Lengkap dengan Hasil Clustering")

    # Buat 2 kolom untuk 2 filter
    col1, col2 = st.columns(2)

    # Filter 1: Klasifikasi Wilayah
    with col1:
        filter_klas = st.multiselect(
            "Filter Klasifikasi Wilayah",
            sorted(df_clustered['klasifikasi_wilayah'].unique()),
            default=sorted(df_clustered['klasifikasi_wilayah'].unique())  # Default: semua terpilih
        )

    # Filter 2: Cluster
    with col2:
        filter_cluster = st.multiselect(
            "Filter Cluster",
            sorted(df_clustered['cluster'].unique()),
            default=sorted(df_clustered['cluster'].unique())
        )

    # Apply filters
    df_filtered = df_clustered[
        (df_clustered['klasifikasi_wilayah'].isin(filter_klas)) &
        (df_clustered['cluster'].isin(filter_cluster))
    ].sort_values('cluster')  # Sort by cluster

    # Define kolom yang akan ditampilkan
    display_cols = [
        'cluster', 'kecamatan', 'jumlah_warga', 'warga_miskin', 'persentase_miskin', 
        'penerima_beasiswa', 'persentase_penerima', 'gap_bantuan', 
        'rasio_beasiswa_kemiskinan', 'persentase_produktif'
    ]

    # Tambahkan kolom bangunan jika tersedia
    if 'total_bangunan' in df_filtered.columns and df_filtered['total_bangunan'].sum() > 0:
        display_cols.extend(['total_bangunan', 'persen_layak'])

    # Tambahkan kolom klasifikasi
    display_cols.extend(['klasifikasi_wilayah', 'status_sasaran'])

    # Filter hanya kolom yang ada
    available_cols = [col for col in display_cols if col in df_filtered.columns]

    # Tampilkan dataframe
    st.dataframe(
        df_filtered[available_cols],
        use_container_width=True,  # Full width
        height=400  # Fixed height dengan scroll
    )

    # Section Export
    st.markdown("### Export Data")
    col1, col2 = st.columns(2)

    # Tombol download CSV
    with col1:
        csv = df_filtered[available_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data (CSV)",
            data=csv,
            file_name='hasil_clustering_kecamatan.csv',
            mime='text/csv',
        )

    # Metric jumlah data
    with col2:
        st.metric("Total Data Ditampilkan", len(df_filtered))

# ============================================================================
# TAB 6: INSIGHTS & REKOMENDASI
# ============================================================================

with tabs[5]:
    st.subheader("Insights dan Rekomendasi Kebijakan")

    st.markdown("### Ringkasan Analisis")

    # KPI Row: 4 metrics ringkasan
    cols = st.columns(4)

    # Metric 1: Jumlah Wilayah Menengah Bawah
    with cols[0]:
        menengah_bawah = len(df_clustered[df_clustered['klasifikasi_wilayah'] == 'Menengah Bawah'])
        cols[0].metric("Wilayah Menengah Bawah", menengah_bawah)

    # Metric 2: Rata-rata Rasio dengan status
    with cols[1]:
        avg_rasio = df_clustered['rasio_beasiswa_kemiskinan'].mean()
        status_rasio = "Baik" if avg_rasio > 0.5 else "Perlu Perhatian"
        cols[1].metric("Rata-rata Rasio", f"{avg_rasio:.2f}", status_rasio)

    # Metric 3: Total Gap
    with cols[2]:
        total_gap = df_clustered['gap_bantuan'].sum()
        cols[2].metric("Total Gap", f"{total_gap:.1f}%")

    # Metric 4: Rata-rata Produktif
    with cols[3]:
        avg_produktif = df_clustered['persentase_produktif'].mean()
        cols[3].metric("Rata-rata Produktif", f"{avg_produktif:.1f}%")

    st.markdown("---")
    st.markdown("### Rekomendasi Berbasis Clustering")

    # REKOMENDASI 1: Wilayah Prioritas Tinggi
    prioritas = df_clustered[df_clustered['status_sasaran'] == 'Tidak Sesuai - Perlu Prioritas']
    if len(prioritas) > 0:
        with st.expander("Wilayah Prioritas Tinggi", expanded=True):
            wilayah_list = ', '.join(sorted(prioritas['kecamatan'].values))
            st.error(f"**{len(prioritas)} Kecamatan:** {wilayah_list}")
            st.markdown("""
            **Rekomendasi Aksi:**
            - Tingkatkan alokasi beasiswa minimal 2x lipat
            - Sosialisasi program bantuan lebih intensif
            - Monitoring penyaluran bantuan secara berkala
            - Program pendampingan khusus untuk warga miskin
            """)

    # REKOMENDASI 2: Wilayah Perlu Peningkatan
    kurang_sesuai = df_clustered[df_clustered['status_sasaran'] == 'Kurang Sesuai - Perlu Ditingkatkan']
    if len(kurang_sesuai) > 0:
        with st.expander(f"Wilayah Perlu Peningkatan ({len(kurang_sesuai)} kecamatan)"):
            wilayah_list = ', '.join(sorted(kurang_sesuai['kecamatan'].values[:10]))
            if len(kurang_sesuai) > 10:
                wilayah_list += "..."
            st.warning(f"**Kecamatan:** {wilayah_list}")
            st.markdown("""
            **Rekomendasi:**
            - Review dan tingkatkan alokasi bertahap
            - Evaluasi kriteria penerima beasiswa
            - Sinkronisasi dengan program kemiskinan lainnya
            """)

    # REKOMENDASI 3: Wilayah Sudah Sesuai
    baik = df_clustered[df_clustered['status_sasaran'].isin(['Baik', 'Cukup Sesuai'])]
    if len(baik) > 0:
        with st.expander(f"Wilayah Sudah Sesuai Sasaran ({len(baik)} kecamatan)"):
            wilayah_list = ', '.join(sorted(baik['kecamatan'].values[:10]))
            if len(baik) > 10:
                wilayah_list += "..."
            st.success(f"**Kecamatan:** {wilayah_list}")
            st.markdown("""
            **Rekomendasi:**
            - Pertahankan alokasi saat ini
            - Monitoring rutin untuk sustainability
            - Jadikan best practice untuk wilayah lain
            """)

    # Metodologi
    st.markdown("---")
    st.markdown("### Metodologi Penelitian")

    with st.expander("Tentang Analisis Ini"):
        st.markdown("""
        **Metode Analisis:**
        - Algoritma: K-Means Clustering (Unsupervised Learning)
        - Evaluasi: Silhouette Score untuk mengukur kualitas clustering
        - Preprocessing: StandardScaler untuk normalisasi fitur
        - Visualisasi: Plotly untuk visualisasi interaktif

        **Dataset:**
        - Data Warga Miskin per Kecamatan
        - Data Penerima Beasiswa per Kecamatan
        - Data Demografi (Usia Produktif, Anak, Lansia)
        - Data Infrastruktur Bangunan (opsional)

        **Tujuan Analisis:**
        1. Segmentasi kecamatan berdasarkan karakteristik sosial-ekonomi
        2. Identifikasi wilayah yang membutuhkan prioritas bantuan
        3. Evaluasi kesesuaian distribusi beasiswa dengan tingkat kemiskinan
        4. Memberikan rekomendasi kebijakan berbasis data

        **Tools:** Python, Streamlit, Scikit-learn, Pandas, Plotly
        """)

# ============================================================================
# SIDEBAR INFO
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Informasi Dashboard")
st.sidebar.info("""
Dashboard Analisis Clustering untuk segmentasi wilayah kecamatan di Surabaya berdasarkan karakteristik sosial-ekonomi dan infrastruktur.

**Metode:** K-Means Clustering

**Data:** 31 Kecamatan Surabaya
""")

# ============================================================================
# END OF CODE
# ============================================================================