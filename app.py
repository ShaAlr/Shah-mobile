"""
====================================================================
SISTEM REKOMENDASI GAME STEAM - Content-Based Filtering
====================================================================
Kelompok 12 - Proyek Sains Data ITS
Ghalib Ibrahim Zardy (5052231028)
M Shah Aquilla Febryano (5052231043)

Catatan Teknis Dataset (games.csv dari Kaggle Steam Dataset):
  Karena AppID tidak bertindak sebagai header kolom, pandas tanpa
  index_col=False menggeser seluruh kolom satu posisi ke kanan.
  Mapping kolom yang sudah diverifikasi manual (menggunakan index_col=False):
    "Tags"              -> Genre aktual game (Action, RPG, dll.)
    "Mac"               -> Flag is_windows (True/False)
    "Linux"             -> Flag is_mac (True/False)
    "Metacritic score"  -> Flag is_linux (True/False)
    "Negative"          -> Jumlah review positif (aktual)
    "Score rank"        -> Jumlah review negatif (aktual)
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Game Recommender | Kelompok 12",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Modern Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0a0a1e 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1a1f2e 100%);
    border-right: 1px solid #2563eb33;
}
.hero-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #7c3aed 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(37,99,235,0.3);
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 800; color: white; margin: 0; }
.hero-banner p  { color: #bfdbfe; margin: 0.3rem 0 0; font-size: 1rem; }
.game-card {
    background: linear-gradient(135deg, #1e293b, #1a2744);
    border: 1px solid #2563eb44; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.game-title { font-size: 1.12rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.6rem; }
.game-meta  { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.4rem; }
.badge { padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.78rem;
         font-weight: 600; display: inline-block; }
.badge-genre  { background:#1e40af; color:#bfdbfe; }
.badge-price  { background:#065f46; color:#a7f3d0; }
.badge-rating { background:#78350f; color:#fde68a; }
.badge-sim    { background:#581c87; color:#e9d5ff; }
.badge-os     { background:#1f2937; color:#9ca3af; }
.sim-bar-wrap { background:#0f172a; border-radius:6px; height:6px; width:100%;
                margin-top:0.6rem; overflow:hidden; }
.sim-bar      { height:100%; border-radius:6px; }
.section-hdr  { font-size:1.3rem; font-weight:700; color:#93c5fd;
                border-left:4px solid #2563eb; padding-left:0.8rem;
                margin: 1.5rem 0 1rem; }
.no-result    { text-align:center; padding:3rem; color:#64748b; font-size:1.1rem;
                background:#1e293b; border-radius:12px; border:1px dashed #334155; }
.footer       { text-align:center; padding:1.5rem; color:#64748b; font-size:0.85rem;
                border-top:1px solid #1e293b; margin-top:3rem; }
.footer span  { color:#60a5fa; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATASET PATH RESOLVER + AUTO-DOWNLOAD
# ─────────────────────────────────────────────

# URL dataset di Hugging Face milik shah0
HF_DATASET_URL = "https://huggingface.co/datasets/shah0/steam-games/resolve/main/games.csv"

LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games.csv")

POSSIBLE_PATHS = [
    "/mnt/user-data/uploads/games.csv",
    LOCAL_CSV,
    "games.csv",
]

def download_dataset(url: str, dest: str) -> bool:
    """Download dataset dari Hugging Face dengan progress bar."""
    import urllib.request
    try:
        bar = st.progress(0, text="⬇️ Mengunduh dataset dari Hugging Face... harap tunggu")
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size, 1.0)
                done_mb  = count * block_size / 1_000_000
                total_mb = total_size / 1_000_000
                bar.progress(pct, text=f"⬇️ Mengunduh: {done_mb:.1f} MB / {total_mb:.1f} MB")
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
        bar.empty()
        return True
    except Exception as e:
        st.error(f"❌ Gagal download: {e}")
        return False

def find_dataset():
    # Cek lokal dulu
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    # Tidak ada lokal → download otomatis dari Hugging Face
    st.info("📥 Dataset tidak ditemukan secara lokal. Mengunduh dari Hugging Face...")
    if download_dataset(HF_DATASET_URL, LOCAL_CSV):
        return LOCAL_CSV
    return None


# ─────────────────────────────────────────────
# DATA LOADING — @st.cache_data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Memuat dataset Steam Games (optimasi memori aktif)...")
def load_and_preprocess(filepath: str, sample_size: int = 56000) -> pd.DataFrame:
    """
    Memuat & memproses dataset dengan strategi efisiensi memori.

    TEKNIS:
    - usecols: hanya 10 dari 39 kolom → hemat RAM ~74%
    - dtype=str: hindari inferensi tipe yang lambat di file besar
    - Smart sampling: 15k game dengan total review tertinggi
      (proxy untuk kualitas data dan relevansi rekomendasi)

    PERGESERAN KOLOM:
    Karena struktur CSV Kaggle Steam Games, kolom-kolom tergeser
    satu posisi. Mapping yang sudah diverifikasi manual (lihat docstring modul).
    """
    needed_cols = [
        "Name", "Price", "Release date",
        "Tags",             # <- Genre aktual
        "Mac",              # <- is_windows
        "Linux",            # <- is_mac
        "Metacritic score", # <- is_linux
        "Negative",         # <- review positif (aktual)
        "Score rank",       # <- review negatif (aktual)
        "Developers",
    ]

    df = pd.read_csv(
        filepath,
        usecols=needed_cols,
        dtype=str,
        low_memory=False,
        index_col=False   # krusial: jangan jadikan AppID sebagai index
    )

    # Rename ke nama semantik yang benar
    df = df.rename(columns={
        "Tags":             "Genres",
        "Mac":              "_win_raw",
        "Linux":            "_mac_raw",
        "Metacritic score": "_linux_raw",
        "Negative":         "_pos_raw",
        "Score rank":       "_neg_raw",
    })

    # Konversi numerik
    df["Price"]      = pd.to_numeric(df["Price"],      errors="coerce").fillna(0.0)
    df["review_pos"] = pd.to_numeric(df["_pos_raw"],   errors="coerce").fillna(0)
    df["review_neg"] = pd.to_numeric(df["_neg_raw"],   errors="coerce").fillna(0)

    # Rating Steam-style: % review positif
    total             = df["review_pos"] + df["review_neg"]
    df["Total_Reviews"] = total
    df["Rating"]      = np.where(total > 0, (df["review_pos"] / total * 100).round(1), 0.0)

    # OS booleans
    df["is_windows"] = df["_win_raw"].str.strip().str.lower().isin(["true", "1", "yes"])
    df["is_mac"]     = df["_mac_raw"].str.strip().str.lower().isin(["true", "1", "yes"])
    df["is_linux"]   = df["_linux_raw"].str.strip().str.lower().isin(["true", "1", "yes"])

    # Bersihkan teks
    df["Genres"]     = df["Genres"].fillna("").str.strip()
    df["Name"]       = df["Name"].fillna("Unknown").str.strip()
    df["Developers"] = df["Developers"].fillna("Unknown").str.strip()

    # Hapus game tanpa genre (tidak bisa di-vectorize)
    df = df[df["Genres"] != ""].copy()

    # Feature untuk TF-IDF: genre dinormalisasi koma -> spasi
    df["content_features"] = df["Genres"].str.replace(",", " ", regex=False)

    # Smart sampling: game paling banyak ulasannya → data paling lengkap
    df_out = (
        df.sort_values("Total_Reviews", ascending=False)
          .drop_duplicates(subset=["Name"])
          .head(sample_size)
          .reset_index(drop=True)
    )

    # Buang kolom temporary
    drop_cols = ["_win_raw","_mac_raw","_linux_raw","_pos_raw","_neg_raw",
                 "review_pos","review_neg"]
    df_out.drop(columns=[c for c in drop_cols if c in df_out.columns], inplace=True)

    return df_out


# ─────────────────────────────────────────────
# TF-IDF & SIMILARITY ENGINE — @st.cache_resource
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Membangun TF-IDF Matrix & Cosine Similarity Engine...")
def build_engine(_df: pd.DataFrame):
    """
    Bangun model Content-Based Filtering.

    @st.cache_resource (bukan cache_data) karena:
    - Objek sklearn tidak serializable secara efisien
    - Satu instance di-share semua user session → hemat RAM signifikan

    TF-IDF Vectorization:
    - max_features=3000: batasi ukuran vocabulary
    - ngram_range=(1,2): tangkap "Action" dan "Action Adventure"
    - sublinear_tf=True: log(TF) agar genre yang sangat sering tidak mendominasi
    - min_df=2: abaikan genre yang hanya ada di 1 game (noise)

    Cosine Similarity:
    - cos(A,B) = (A·B) / (|A|×|B|)  → range [0,1]
    - Lebih fair vs Euclidean untuk teks sparse
    """
    vec = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word"
    )
    tfidf_mat = vec.fit_transform(_df["content_features"])
    # CATATAN: Tidak pre-compute full similarity matrix (n² memory = crash)
    # Similarity dihitung real-time per query: user_vector vs semua game = O(n) RAM
    return vec, tfidf_mat


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def get_recommendations(df, vectorizer, tfidf_mat,
                        selected_genres, budget_usd,
                        target_os, top_n, include_free):
    """
    4-tahap Content-Based Filtering:
    1. Build user profile vector dari genre pilihan user
    2. Cosine similarity: profil user vs semua game (operasi matrix)
    3. Hard constraint filtering: budget + OS + free/paid
    4. Ranking Top-N berdasarkan similarity score tertinggi
    """
    # Tahap 1: User query → TF-IDF vector
    user_vec  = vectorizer.transform([" ".join(selected_genres)])

    # Tahap 2: Similarity scores (1 × n_games)
    scores = cosine_similarity(user_vec, tfidf_mat).flatten()

    result = df.copy()
    result["similarity_score"] = scores

    # Tahap 3: Hard filters
    result = result[result["Price"] <= budget_usd]
    if not include_free:
        result = result[result["Price"] > 0]
    os_map = {"Windows": "is_windows", "Mac": "is_mac", "Linux": "is_linux"}
    if target_os in os_map:
        result = result[result[os_map[target_os]] == True]

    # Tahap 4: Rank & Top-N
    result = (
        result
        .sort_values("similarity_score", ascending=False)
        .drop_duplicates(subset=["Name"])
        .query("similarity_score > 0.01")
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def render_game_card(idx, row):
    sim_pct   = int(row["similarity_score"] * 100)
    bar_color = "#22c55e" if sim_pct >= 70 else ("#f59e0b" if sim_pct >= 40 else "#ef4444")
    match_lbl = "🔥 Sangat Relevan" if sim_pct >= 70 else ("✅ Relevan" if sim_pct >= 40 else "💡 Terkait")
    price_str = "🆓 GRATIS" if row["Price"] == 0 else f"💰 ${row['Price']:.2f}"
    rating_str = f"⭐ {row['Rating']:.0f}%" if row["Total_Reviews"] >= 5 else "⭐ N/A"
    os_parts = []
    if row.get("is_windows"): os_parts.append("🪟Win")
    if row.get("is_mac"):     os_parts.append("🍎Mac")
    if row.get("is_linux"):   os_parts.append("🐧Linux")
    os_str = " · ".join(os_parts) if os_parts else "—"
    genres_list  = [g.strip() for g in row["Genres"].split(",") if g.strip()]
    genres_disp  = " · ".join(genres_list[:4])
    if len(genres_list) > 4: genres_disp += f" +{len(genres_list)-4}"

    st.markdown(f"""
    <div class="game-card">
        <div class="game-title">#{idx+1} &nbsp;&nbsp; {row['Name']}</div>
        <div class="game-meta">
            <span class="badge badge-sim">{match_lbl} ({sim_pct}%)</span>
            <span class="badge badge-price">{price_str}</span>
            <span class="badge badge-rating">{rating_str}</span>
            <span class="badge badge-genre">🎮 {genres_disp}</span>
            <span class="badge badge-os">{os_str}</span>
        </div>
        <div class="sim-bar-wrap">
            <div class="sim-bar" style="width:{sim_pct}%;
                 background:linear-gradient(90deg,#2563eb,{bar_color});"></div>
        </div>
        <div style="color:#475569;font-size:0.72rem;margin-top:0.4rem;">
            Cosine Similarity: {row['similarity_score']:.4f} &nbsp;|&nbsp;
            Reviews: {int(row['Total_Reviews']):,}
        </div>
    </div>""", unsafe_allow_html=True)


def render_eda(df):
    st.markdown('<div class="section-hdr">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(f"Analisis deskriptif **{len(df):,} game** dari Kaggle Steam Games Dataset.")

    # Metrics
    total     = len(df)
    free_pct  = (df["Price"] == 0).sum() / total * 100
    avg_price = df[df["Price"] > 0]["Price"].mean()
    rated     = df[df["Total_Reviews"] >= 10]
    avg_rating= rated["Rating"].mean() if len(rated) else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🎮 Total Game", f"{total:,}")
    c2.metric("🆓 Game Gratis", f"{free_pct:.1f}%")
    c3.metric("💵 Harga Avg", f"${avg_price:.2f}")
    c4.metric("⭐ Rating Avg", f"{avg_rating:.1f}%")
    c5.metric("🪟 Windows", f"{df['is_windows'].sum():,}")
    st.divider()

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏷️ Top 15 Genre")
        gc = (df["Genres"].str.split(",").explode().str.strip()
              .value_counts().head(15))
        fig = px.bar(x=gc.values, y=gc.index, orientation="h",
                     color=gc.values, color_continuous_scale="Blues",
                     template="plotly_dark",
                     labels={"x":"Jumlah Game","y":"Genre"})
        fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                          coloraxis_showscale=False, height=420,
                          margin=dict(l=0,r=20,t=20,b=0))
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} game<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Distribusi Harga (≤ $80)")
        pd_df = df[(df["Price"]>0) & (df["Price"]<=80)]
        fig = px.histogram(pd_df, x="Price", nbins=40,
                           color_discrete_sequence=["#3b82f6"],
                           template="plotly_dark",
                           labels={"Price":"Harga (USD)","count":"Jumlah"})
        med = pd_df["Price"].median()
        fig.add_vline(x=med, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Median: ${med:.2f}",
                      annotation_font_color="#f59e0b",
                      annotation_position="top right")
        fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                          height=420, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("⭐ Distribusi Rating (min. 10 ulasan)")
        r_df = df[df["Total_Reviews"] >= 10]
        fig = px.histogram(r_df, x="Rating", nbins=25,
                           color_discrete_sequence=["#8b5cf6"],
                           template="plotly_dark",
                           labels={"Rating":"Rating Positif (%)","count":"Jumlah"})
        fig.add_vline(x=r_df["Rating"].mean(), line_dash="dash", line_color="#22c55e",
                      annotation_text=f"Mean: {r_df['Rating'].mean():.1f}%",
                      annotation_font_color="#22c55e",
                      annotation_position="top left")
        fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                          height=360, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("🖥️ Kompatibilitas OS")
        os_df = pd.DataFrame({
            "Platform": ["Windows","Mac","Linux"],
            "Jumlah":   [df["is_windows"].sum(), df["is_mac"].sum(), df["is_linux"].sum()]
        })
        fig = px.pie(os_df, names="Platform", values="Jumlah",
                     color_discrete_sequence=["#2563eb","#10b981","#f59e0b"],
                     template="plotly_dark", hole=0.5)
        fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                          height=360, margin=dict(l=0,r=0,t=20,b=0))
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          hovertemplate="<b>%{label}</b><br>%{value:,} game<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Scatter
    st.subheader("🔬 Harga vs Rating (berbayar, min. 100 ulasan)")
    sc_df = df[(df["Price"]>0)&(df["Price"]<=60)&(df["Total_Reviews"]>=100)]
    if len(sc_df) >= 10:
        fig = px.scatter(sc_df, x="Price", y="Rating",
                         color="Rating", size="Total_Reviews", size_max=22,
                         hover_name="Name",
                         color_continuous_scale="RdYlGn",
                         template="plotly_dark", opacity=0.72,
                         labels={"Price":"Harga (USD)","Rating":"Rating Positif (%)"})
        fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#1e293b", height=450)
        fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>$%{x:.2f} | %{y:.1f}%<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📚 Detail Teknis Algoritma"):
        st.markdown(f"""
        **Preprocessing:**
        - `usecols` → hanya 10/39 kolom dibaca (~74% hemat memori)
        - Seluruh data digunakan: {sample_size:,} game (semua game yang memiliki genre valid)
        - Rating = `Positif / (Positif + Negatif) × 100%`

        **TF-IDF Parameters:**
        - `max_features=3000` | `ngram_range=(1,2)` | `min_df=2` | `sublinear_tf=True`

        **Cosine Similarity:**
        `cos(A,B) = (A·B) / (|A|×|B|)` — range [0,1]

        **Caching:**
        - `@st.cache_data`: data preprocessing (pickle-able)
        - `@st.cache_resource`: TF-IDF matrix (shared, non-serializable)
        """)


sample_size = 56000   # semua data yang tersedia setelah filtering


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.8rem;">🎮</div>
        <div style="font-size:1.05rem;font-weight:800;color:#60a5fa;">Steam Recommender</div>
        <div style="font-size:0.72rem;color:#475569;margin-top:0.2rem;">
            TF-IDF · Cosine Similarity · CBF
        </div>
    </div>""", unsafe_allow_html=True)
    st.sidebar.divider()

    st.sidebar.markdown("**🎯 Preferensi Genre**")
    all_g = sorted(set(
        g.strip() for genres in df["Genres"]
        for g in genres.split(",") if g.strip() and len(g.strip()) > 1
    ))
    selected = st.sidebar.multiselect(
        "Pilih genre favorit:", options=all_g,
        default=["Action", "Adventure"],
        help="Pilih 1+ genre. Sistem menemukan game paling mirip."
    )
    st.sidebar.divider()

    st.sidebar.markdown("**💰 Batas Anggaran**")
    include_free = st.sidebar.checkbox("Sertakan game gratis (F2P)", value=True)
    budget = st.sidebar.slider("Budget Maksimum (USD):", 0, 100, 30, 1, format="$%d")
    st.sidebar.divider()

    st.sidebar.markdown("**🖥️ Sistem Operasi**")
    target_os = st.sidebar.selectbox("Platform:", ["Any","Windows","Mac","Linux"])
    st.sidebar.divider()

    st.sidebar.markdown("**📋 Jumlah Rekomendasi**")
    top_n = st.sidebar.slider("Tampilkan Top-N:", 5, 30, 10, 5)
    st.sidebar.divider()

    run_btn = st.sidebar.button("🚀 Cari Rekomendasi", use_container_width=True, type="primary")

    st.sidebar.markdown(f"""
    <div style="background:#0f172a;border-radius:8px;padding:0.8rem;
                border:1px solid #1e3a8a;margin-top:0.5rem;">
        <div style="font-size:0.72rem;color:#475569;">📦 Dataset Dimuat</div>
        <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.2rem;">
            <b style="color:#60a5fa;">{len(df):,}</b> game<br>
            Kaggle Steam Games Dataset
        </div>
    </div>""", unsafe_allow_html=True)

    return selected, budget, target_os, top_n, run_btn, include_free


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero-banner">
        <h1>🎮 Sistem Rekomendasi Game Steam</h1>
        <p>Content-Based Filtering · TF-IDF Vectorization · Cosine Similarity</p>
        <p style="margin-top:0.4rem;font-size:0.82rem;color:#93c5fd;">
            Kelompok 12 &nbsp;|&nbsp; Proyek Sains Data &nbsp;|&nbsp;
            Institut Teknologi Sepuluh Nopember (ITS)
        </p>
    </div>""", unsafe_allow_html=True)

    # Load dataset
    path = find_dataset()
    if path is None:
        st.error("❌ File `games.csv` tidak ditemukan! Letakkan di direktori yang sama dengan app.py.")
        st.stop()

    df         = load_and_preprocess(path, sample_size=sample_size)
    vec, tmat = build_engine(df)

    # Sidebar
    selected, budget, target_os, top_n, run_btn, include_free = render_sidebar(df)

    # Tabs
    tab1, tab2 = st.tabs(["🎯 Rekomendasi Game", "📊 Data Insights (EDA)"])

    # ── TAB 1 ──
    with tab1:
        if not selected:
            st.info("👈 Pilih genre di sidebar untuk memulai rekomendasi.")
        else:
            with st.spinner("🔍 Menghitung similarity scores..."):
                recs = get_recommendations(
                    df, vec, tmat, selected, budget,
                    target_os, top_n, include_free
                )

            if recs.empty:
                st.markdown("""
                <div class="no-result">
                    😕 <b>Tidak ada game yang memenuhi semua kriteria.</b><br><br>
                    Coba: perluas budget · ubah genre · pilih OS "Any"
                </div>""", unsafe_allow_html=True)
            else:
                genre_str = " · ".join(selected)
                bstr = f"${budget} {'(+Gratis)' if include_free else '(Berbayar)'}"
                st.markdown(f"""
                <div class="section-hdr">✅ {len(recs)} Rekomendasi Ditemukan</div>
                <p style="color:#94a3b8;margin:-0.5rem 0 1.5rem;font-size:0.9rem;">
                    🎮 <b style="color:#60a5fa;">{genre_str}</b> &nbsp;·&nbsp;
                    💰 <b style="color:#34d399;">{bstr}</b> &nbsp;·&nbsp;
                    🖥️ <b style="color:#a78bfa;">{target_os}</b>
                </p>""", unsafe_allow_html=True)

                for i, (_, row) in enumerate(recs.iterrows()):
                    render_game_card(i, row)

                # Mini chart
                if len(recs) >= 3:
                    st.divider()
                    st.subheader("📈 Similarity Score — Distribusi Hasil")
                    fig = px.bar(
                        recs, x="similarity_score", y="Name", orientation="h",
                        color="similarity_score", color_continuous_scale="Blues",
                        template="plotly_dark",
                        labels={"similarity_score":"Cosine Similarity","Name":"Game"}
                    )
                    fig.update_layout(
                        plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
                        coloraxis_showscale=False,
                        height=max(300, len(recs)*36),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0,r=0,t=20,b=0)
                    )
                    fig.update_traces(hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>")
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📋 Tabel Data Rekomendasi"):
                        tbl = recs[["Name","Genres","Price","Rating",
                                    "Total_Reviews","similarity_score"]].copy()
                        tbl.columns = ["Game","Genre","Harga (USD)","Rating (%)",
                                       "Total Ulasan","Similarity Score"]
                        tbl["Harga (USD)"] = tbl["Harga (USD)"].apply(
                            lambda x: "Gratis" if x==0 else f"${x:.2f}")
                        tbl["Similarity Score"] = tbl["Similarity Score"].apply(
                            lambda x: f"{x:.4f}")
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── TAB 2 ──
    with tab2:
        render_eda(df)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Presented by
            <span>Ghalib Ibrahim Zardy (5052231028)</span> &amp;
            <span>M Shah Aquilla Febryano (5052231043)</span>
            — <span>Kelompok 12</span>
        </p>
        <p style="margin-top:0.3rem;font-size:0.78rem;">
            Proyek Sains Data · Institut Teknologi Sepuluh Nopember (ITS) ·
            Dataset: Kaggle Steam Games
        </p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
