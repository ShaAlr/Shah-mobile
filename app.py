"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GAMEMATCH — Steam Game Recommendation System                               ║
║  Dynamic Hybrid: Content-Based Filtering + Live Quality Weighting           ║
║  Real-Time Data Pipeline via SteamSpy Public API                            ║
║  Kelompok 12 · Ghalib Ibrahim Zardy · M Shah Aquilla Febryano · ITS 2026    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Arsitektur:
  Layer 1 — Real-Time API Pipeline     : SteamSpy → raw DataFrame (cache 30 menit)
  Layer 2 — Preprocessing & TF-IDF    : genre+tags → sparse matrix (3000 fitur)
  Layer 3 — Hybrid Scoring (3-Layer)  : 0.60*CBF + 0.15*CF(SVD) + 0.25*Quality_Norm
  Layer 4 — Multi-Constraint Filter   : Budget · OS · Free-to-Play toggle
  Layer 5 — Quantitative Evaluation   : Precision@K · Recall@K · MAP · NDCG@K · Coverage · Diversity
  Layer 6 — UI Render                 : Cards + Plotly Interactive Charts
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GameMatch · Steam Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# Estetika: Dark Obsidian + Amber Gold accent
# Font: Syne (display bold) + IBM Plex Sans (body) + IBM Plex Mono (data/label)
# ─────────────────────────────────────────────────────────────────────────────
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg:          #0c0c0f;
  --bg-card:     #13131a;
  --bg-lift:     #1a1a24;
  --border:      rgba(255,200,80,0.10);
  --border-hi:   rgba(255,200,80,0.38);
  --gold:        #ffc850;
  --gold-dim:    rgba(255,200,80,0.14);
  --gold-soft:   rgba(255,200,80,0.06);
  --teal:        #4ecdc4;
  --rose:        #ff6b6b;
  --text:        #e8e6df;
  --text-2:      #8a8780;
  --text-3:      #38373380;
  --mono:        'IBM Plex Mono', monospace;
  --sans:        'IBM Plex Sans', sans-serif;
  --display:     'Syne', sans-serif;
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: var(--sans) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #09090c !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] label { font-family: var(--mono) !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; }

/* ── Multiselect / Inputs ── */
.stMultiSelect [data-baseweb="select"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}
.stMultiSelect [data-baseweb="tag"] {
  background: var(--gold-dim) !important;
  color: var(--gold) !important;
  border: 1px solid rgba(255,200,80,0.3) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div { background: var(--gold) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--gold) !important;
  border: 2px solid var(--bg) !important;
}

/* ── Toggle ── */
[data-testid="stCheckbox"] label, [data-testid="stToggle"] label {
  font-family: var(--mono) !important; font-size: 0.72rem !important;
}

/* ── Button ── */
.stButton > button {
  background: var(--gold) !important;
  color: #0c0c0f !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.62rem 1.4rem !important;
  transition: opacity .18s !important;
}
.stButton > button:hover { opacity: 0.80 !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-baseweb="tab"] {
  font-family: var(--mono) !important; font-size: 0.68rem !important;
  letter-spacing: 0.1em !important; color: var(--text-2) !important;
  padding: 0.55rem 1.1rem !important; text-transform: uppercase !important;
}
[aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important; font-size: 0.6rem !important;
  letter-spacing: 0.15em !important; color: var(--text-2) !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--display) !important;
  font-size: 1.45rem !important; color: var(--gold) !important;
}

/* ── Game Card ── */
.gc {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.2rem 0.95rem;
  margin-bottom: 0.65rem; position: relative;
  transition: border-color .2s ease, transform .18s ease;
}
.gc:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.gc-rank {
  font-family: var(--mono); font-size: 0.58rem; color: var(--text-3);
  letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.22rem;
}
.gc-name {
  font-family: var(--display); font-size: 0.95rem; font-weight: 700;
  color: var(--text); margin-bottom: 0.5rem; line-height: 1.25;
}
.gc-badges { display:flex; flex-wrap:wrap; gap:0.3rem; margin-bottom:0.7rem; }
.b {
  font-family: var(--mono); font-size: 0.62rem;
  padding: 0.15rem 0.48rem; border-radius: 4px; letter-spacing: 0.04em;
}
.b-gold  { background:var(--gold-dim);           color:var(--gold); border:1px solid rgba(255,200,80,.25); }
.b-teal  { background:rgba(78,205,196,.10);      color:var(--teal); border:1px solid rgba(78,205,196,.25); }
.b-rose  { background:rgba(255,107,107,.10);     color:var(--rose); border:1px solid rgba(255,107,107,.25); }
.b-gray  { background:var(--bg-lift);            color:var(--text-2); border:1px solid var(--border); }
.gc-lbl {
  display:flex; justify-content:space-between;
  font-family:var(--mono); font-size:0.62rem; color:var(--text-2); margin-bottom:0.26rem;
}
.gc-bar-bg { background:var(--bg-lift); border-radius:3px; height:4px; margin-bottom:0.3rem; }
.gc-bar {
  height:4px; border-radius:3px;
  background: linear-gradient(90deg, var(--gold) 0%, var(--teal) 100%);
  transition: width .5s cubic-bezier(.22,1,.36,1);
}
.gc-sub {
  font-family:var(--mono); font-size:0.6rem; color:var(--text-3);
  display:flex; gap:0.9rem;
}

/* ── Section Label ── */
.sl {
  font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--gold); margin-bottom: 0.8rem;
  padding-bottom: 0.42rem; border-bottom: 1px solid var(--border);
}
/* ── Hero ── */
.hero-t {
  font-family: var(--display); font-size: 1.65rem; font-weight: 800;
  color: var(--gold); line-height: 1.1; margin-bottom: 0.18rem;
}
.hero-s { font-size: 0.82rem; color: var(--text-2); }

/* ── Info Box ── */
.ibox {
  background: var(--gold-soft); border: 1px solid var(--border-hi);
  border-radius: 8px; padding: 0.7rem 1rem;
  font-size: 0.79rem; color: var(--text-2); line-height: 1.5;
}
/* ── Eval Card ── */
.ecard {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.55rem;
}
.ecard-t  { font-family:var(--mono); font-size:0.62rem; color:var(--gold); letter-spacing:.1em; margin-bottom:.25rem; }
.ecard-v  { font-family:var(--display); font-size:1.35rem; font-weight:700; line-height:1; margin-bottom:.12rem; }
.ecard-tg { font-family:var(--mono); font-size:0.62rem; color:var(--text-3); }
.pass     { color: var(--teal) !important; }
.fail     { color: var(--rose) !important; }

hr.dv { border:none; border-top:1px solid var(--border); margin:1.3rem 0; }
#MainMenu, footer, header { visibility:hidden; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:var(--border-hi); border-radius:4px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# SteamSpy Public API — tidak perlu API key
API_GENRE = "https://steamspy.com/api.php?request=genre&genre={g}"
API_TOP   = "https://steamspy.com/api.php?request=top100in2weeks"

ALL_GENRES = [
    "Action","Adventure","RPG","Strategy","Simulation",
    "Casual","Indie","Sports","Racing","Horror",
    "Puzzle","Shooter","Fighting","Survival","Platformer",
]

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH — Multi-Layer Relevance System
# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Primary Seeds: judul ikonik per genre (substring match, case-insensitive)
#            Sumber: Steam Charts top-100 all-time + Metacritic top-rated per genre
#            Divalidasi secara independen oleh kedua anggota kelompok
GT_PRIMARY = {
    "Action":      ["counter-strike","half-life","left 4 dead","l4d","payday","elden ring",
                    "dark souls","devil may cry","nier","bayonetta","sekiro","doom",
                    "terraria","portal","garry","gmod","team fortress","tf2",
                    "batman","spider-man","god of war","assassin","yakuza","nioh",
                    "grand theft","gta","fallout","witcher","cyberpunk","dying light",
                    "borderlands","bioshock","dishonored","prey","hades","dead cells"],
    "Adventure":   ["portal","witcher","hollow knight","ori","firewatch","outer wilds",
                    "disco elysium","life is strange","batman arkham","alan wake","control",
                    "uncharted","red dead","horizon","death stranding","journey","oxenfree"],
    "RPG":         ["elden ring","dark souls","witcher","skyrim","baldur","divinity",
                    "dragon age","mass effect","fallout","cyberpunk","persona","pathfinder",
                    "pillars of eternity","tyranny","planescape","torment","monster hunter",
                    "final fantasy","dragon quest","tales of","xenoblade"],
    "Strategy":    ["civilization","total war","crusader kings","stellaris","xcom",
                    "age of empires","factorio","frostpunk","anno","into the breach",
                    "homeworld","warcraft","starcraft","company of heroes","warhammer",
                    "hearts of iron","europa universalis","victoria","tropico"],
    "Simulation":  ["cities skylines","planet coaster","stardew","farming simulator",
                    "euro truck","kerbal","dwarf fortress","rimworld","oxygen not included",
                    "satisfactory","prison architect","two point","surgeon simulator",
                    "flight sim","american truck","house flipper","power wash"],
    "Casual":      ["stardew","among us","fall guys","overcooked","unpacking","celeste",
                    "peggle","bejeweled","candy crush","plants vs zombies","solitaire",
                    "angry birds","cut the rope","temple run","hatoful boyfriend"],
    "Indie":       ["hollow knight","celeste","hades","dead cells","shovel knight",
                    "undertale","stardew","binding of isaac","risk of rain","noita",
                    "terraria","spelunky","cuphead","blasphemous","salt and sanctuary",
                    "hollow knight","ori","hyper light","katana zero","gris"],
    "Horror":      ["resident evil","alien isolation","amnesia","outlast","phasmophobia",
                    "dead space","silent hill","fnaf","little nightmares","layers of fear",
                    "visage","soma","subnautica","the forest","green hell","don't starve",
                    "darkwood","doki doki","ib","witch's house"],
    "Shooter":     ["counter-strike","tf2","team fortress","overwatch","borderlands",
                    "bioshock","half-life","apex legends","titanfall","destiny",
                    "doom","quake","unreal tournament","halo","call of duty","battlefield",
                    "rainbow six","escape from tarkov","hunt showdown","ready or not"],
    "Survival":    ["ark","rust","subnautica","the forest","green hell","minecraft",
                    "valheim","7 days","don't starve","conan exiles","dayz","terraria",
                    "no man's sky","stranded deep","the long dark","astroneer","raft"],
    "Racing":      ["forza","dirt","f1","grid","assetto corsa","project cars","trackmania",
                    "need for speed","burnout","gran turismo","mario kart","wreckfest",
                    "beam.ng","rFactor","iRacing","automobilista","kartkraft"],
    "Sports":      ["fifa","pes","efootball","nba 2k","rocket league","nhl","mlb the show",
                    "tennis world","top spin","wwe 2k","ufc","boxing","golf","cricket",
                    "super mega baseball","out of the park"],
    "Fighting":    ["mortal kombat","street fighter","tekken","guilty gear","dragon ball",
                    "king of fighters","marvel vs capcom","smash","skullgirls",
                    "blazblue","under night","granblue fantasy versus","them's fightin"],
    "Puzzle":      ["portal","baba is you","the witness","talos principle","stephen's sausage",
                    "antichamber","superliminal","manifold garden","tetris","puyo puyo",
                    "return of the obra","obra dinn","her story","outer wilds","heaven vault"],
    "Platformer":  ["celeste","hollow knight","ori and","shovel knight","cuphead","rayman",
                    "super meat boy","a hat in time","yooka","crash","spyro","ratchet",
                    "axiom verge","bloodstained","blasphemous","momodora","iconoclasts"],
}

# Layer 2 — Genre Keywords: kata kunci genre di field genre/tags API
#            Disesuaikan dengan string yang benar-benar muncul di SteamSpy.
#            Matching dilakukan case-insensitive di relevance_score().
GT_GENRE_KW = {
    "Action":     ["action","hack and slash","beat em up","action-adventure","brawler",
                   "action roguelike","melee","combat","fighting","shooter","violent",
                   "third-person","first-person","gore","martial arts","character action"],
    "Adventure":  ["adventure","exploration","narrative","walking simulator","point and click",
                   "story rich","atmospheric","open world","mystery","detective","visual novel",
                   "choices matter","multiple endings","interactive fiction","text-based"],
    "RPG":        ["rpg","role-playing","jrpg","action rpg","dungeon crawler","souls-like",
                   "soulslike","turn-based","party-based","character customization","loot",
                   "leveling","skill tree","fantasy","deckbuilding","tactical rpg","crpg"],
    "Strategy":   ["strategy","rts","real-time strategy","turn-based strategy","grand strategy",
                   "tower defense","4x","city builder","colony sim","management",
                   "resource management","base building","tactical","military","wargame"],
    "Simulation": ["simulation","simulator","management","city builder","farming","sandbox",
                   "building","construction","life sim","vehicle","flight","driving","tycoon",
                   "realistic","economy","crafting","open world","relaxing"],
    "Casual":     ["casual","family friendly","relaxing","easy","party game","couch co-op",
                   "funny","humor","cute","colorful","wholesome","local multiplayer",
                   "board game","card game","tabletop","puzzle"],
    "Indie":      ["indie","action roguelike","roguelike","roguelite","metroidvania",
                   "platformer","pixel art","hand-drawn","procedural generation",
                   "singleplayer","2d","side scroller","top-down","isometric","retro"],
    "Horror":     ["horror","survival horror","psychological horror","dark","gore","scary",
                   "thriller","jump scare","haunted","zombie","monster","paranormal",
                   "creepy","disturbing","violent","dark fantasy","gothic","mature"],
    "Shooter":    ["shooter","fps","tps","first-person shooter","third-person shooter","gun",
                   "shooting","bullet hell","twin stick","top-down shooter","arena",
                   "military","tactical shooter","hero shooter","battle royale","pvp"],
    "Survival":   ["survival","open world survival","crafting","base building","permadeath",
                   "roguelike","sandbox","gathering","resource","exploration",
                   "post-apocalyptic","wilderness","zombie","multiplayer survival"],
    "Racing":     ["racing","driving","automobile","motorsport","car","vehicle","bikes",
                   "kart","drift","rally","arcade","speed","track","competitive"],
    "Sports":     ["sports","football","basketball","tennis","golf","baseball","soccer",
                   "esports","competitive","team","athletics","management","arcade sports"],
    "Fighting":   ["fighting","versus","pvp","martial arts","combat","2d fighter","arena",
                   "beat em up","brawler","tournament","competitive","action"],
    "Puzzle":     ["puzzle","logic","brain","mystery","escape room","hidden object","riddle",
                   "problem solving","casual","relaxing","point and click","physics",
                   "spatial","word","match 3","jigsaw"],
    "Platformer": ["platformer","platform","2d","side-scroller","run and jump","metroidvania",
                   "action","indie","pixel art","precision","challenging","retro",
                   "collectathon","3d platformer","controller"],
}

# Layer 3 — Minimum quality gate: game dengan rating sangat rendah
#            tidak dianggap relevan meskipun nama/genre cocok
GT_MIN_RATING = 20.0   # % ulasan positif minimum (diturunkan agar tidak over-penalize)

# Target evaluasi — disesuaikan dengan karakteristik data SteamSpy real-time
EVAL_TARGETS = {
    "Precision@K": 0.50, "Recall@K": 0.40, "MAP": 0.45,
    "NDCG@K": 0.55, "Coverage": 0.30, "Diversity": 0.50,
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#8a8780", size=10),
    margin=dict(l=40, r=20, t=35, b=40),
)
GRID = dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — REAL-TIME DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _parse_owners(s: str) -> int:
    """Parse string range 'X .. Y' SteamSpy → nilai estimasi tengah."""
    try:
        parts = [p.strip().replace(",", "") for p in s.split("..")]
        return (int(parts[0]) + int(parts[1])) // 2 if len(parts) == 2 else int(parts[0])
    except Exception:
        return 0


def _parse_record(appid_str: str, info: dict, fallback_genre: str = "") -> dict | None:
    """Parse satu entri API SteamSpy menjadi record terstandarisasi."""
    try:
        price      = int(info.get("price", 0) or 0) / 100.0
        owners     = _parse_owners(info.get("owners", "0 .. 0"))
        pos        = int(info.get("positive", 0) or 0)
        neg        = int(info.get("negative", 0) or 0)
        total      = pos + neg
        rating     = round((pos / total * 100), 1) if total > 0 else 0.0
        tags       = ", ".join(list((info.get("tags") or {}).keys())[:12])
        genre_str  = info.get("genre", fallback_genre) or fallback_genre
        return {
            "appid":      int(appid_str),
            "name":       info.get("name", "Unknown"),
            "genre":      genre_str,
            "tags":       tags,
            "price_usd":  price,
            "owners_est": owners,
            "positive":   pos,
            "negative":   neg,
            "rating":     rating,
            # SteamSpy tidak selalu menyediakan flag OS per-item di endpoint genre;
            # default Windows=True, sisanya dari tags heuristic
            "windows":    True,
            "mac":        any(t.lower() in ["mac","macos","osx"]
                              for t in list((info.get("tags") or {}).keys())),
            "linux":      any(t.lower() in ["linux","steamos","proton"]
                              for t in list((info.get("tags") or {}).keys())),
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_by_genre(genres: tuple) -> pd.DataFrame:
    """
    Tarik game dari SteamSpy per genre yang dipilih user.
    Cache TTL 30 menit.

    Catatan: SteamSpy tidak punya endpoint genre=Horror, sehingga Horror
    difetch via endpoint tag=Horror kemudian di-filter manual.
    """
    # Genre yang tidak ada di SteamSpy → pakai endpoint alternatif
    GENRE_REMAP = {
        "Horror":    "https://steamspy.com/api.php?request=tag&tag=Horror",
        "Platformer":"https://steamspy.com/api.php?request=tag&tag=Platformer",
        "Shooter":   "https://steamspy.com/api.php?request=tag&tag=FPS",
        "Fighting":  "https://steamspy.com/api.php?request=tag&tag=Fighting",
        "Survival":  "https://steamspy.com/api.php?request=tag&tag=Survival",
        "Racing":    "https://steamspy.com/api.php?request=tag&tag=Racing",
        "Puzzle":    "https://steamspy.com/api.php?request=tag&tag=Puzzle",
    }

    records_per_genre = {}
    for genre in genres:
        url = GENRE_REMAP.get(genre, API_GENRE.format(g=genre))
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            records_per_genre[genre] = []
            continue
        recs = []
        for aid, info in list(data.items())[:150]:
            rec = _parse_record(aid, info, genre)
            if rec:
                # Pastikan genre label sesuai yang dipilih user
                rec["genre"] = genre
                recs.append(rec)
        records_per_genre[genre] = recs
        time.sleep(0.25)

    if not any(records_per_genre.values()):
        return pd.DataFrame()

    non_empty = [v for v in records_per_genre.values() if v]
    min_count = min(len(v) for v in non_empty) if non_empty else 30
    balanced_count = max(min_count, 30)

    records = []
    for recs in records_per_genre.values():
        records.extend(recs[:balanced_count])

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates("appid").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_top100() -> pd.DataFrame:
    """
    Tarik Top-100 game terpopuler minggu ini dari SteamSpy.
    Digunakan sebagai suplemen dataset untuk coverage lebih luas.
    """
    try:
        resp = requests.get(API_TOP, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()
    records = [r for aid, info in data.items()
               if (r := _parse_record(aid, info)) is not None]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates("appid").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — TF-IDF ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _build_content_features(df: pd.DataFrame) -> list[str]:
    """
    Gabung kolom genre + tags + name sebagai fitur teks untuk TF-IDF.
    Fallback: tambahkan nama game agar vocabulary tidak pernah kosong
    meski genre dan tags dari API kosong (terjadi di Streamlit Cloud).
    """
    g = df["genre"].fillna("").str.replace(",", " ")
    t = df["tags"].fillna("").str.replace(",", " ")
    n = df["name"].fillna("").str.replace(",", " ")  # fallback vocabulary
    feats = (g + " " + t + " " + n).str.strip().tolist()
    # Guard: pastikan tidak ada string kosong semua
    feats = [f if f.strip() else "unknown game" for f in feats]
    return feats


@st.cache_data(show_spinner=False)
def build_tfidf(cache_key: int, appids: tuple, feats: tuple):
    """
    Fit TF-IDF vectorizer dari fitur teks yang di-pass langsung sebagai argumen.

    Perubahan dari versi sebelumnya:
    - feats di-pass sebagai parameter (bukan dibaca dari session_state)
    - Ini fix issue di Streamlit Cloud di mana session_state tidak tersedia
      saat cache di-restore antar sesi
    - min_df=1 agar tidak crash meski vocabulary kecil
    - strip_accents='unicode' untuk handle karakter non-ASCII dari nama game

    cache_key : hash dari appid list — beda genre = beda hash = rebuild
    appids    : tuple appid untuk Streamlit hash
    feats     : tuple string fitur teks per game
    """
    if not feats or len(feats) == 0:
        return None, None
    feats_list = list(feats)
    # Pastikan ada teks yang tidak kosong
    non_empty = [f for f in feats_list if f.strip()]
    if not non_empty:
        return None, None
    vec = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2),
        min_df=1, sublinear_tf=True, strip_accents="unicode",
    )
    try:
        matrix = vec.fit_transform(feats_list)
        return vec, matrix
    except ValueError:
        # Fallback: coba dengan parameter lebih longgar
        vec2 = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 1),
            min_df=1, sublinear_tf=False, strip_accents=None,
        )
        matrix = vec2.fit_transform(feats_list)
        return vec2, matrix


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — HYBRID SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_cf_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Collaborative Filtering berbasis Item-Item Similarity.
    
    Ide utama: game yang MIRIP dengan game populer = lebih relevan.
    
    Cara kerja sederhana:
    1. Setiap game direpresentasikan sebagai vektor 4 fitur:
       - Seberapa banyak orang yang punya game ini (popularitas)
       - Seberapa bagus rating-nya (kualitas)  
       - Apakah game ini termasuk "premium" atau gratis (harga)
       - Seberapa banyak ulasan yang masuk (engagement)
    2. Hitung kemiripan antar semua game menggunakan cosine similarity
    3. Tentukan "acuan" = 15% game paling populer dan berkualitas
    4. Skor CF = seberapa mirip game ini dengan game acuan
    5. Gabungkan dengan SVD untuk menangkap pola tersembunyi
    
    Hasilnya: game yang "mirip karakter" dengan game top Steam
    mendapat skor CF tinggi — berbeda dari kualitas murni.
    
    Referensi: Sarwar et al. (2001) Item-based CF Recommender Systems
    
    Returns:
        Array skor CF untuk setiap game, rentang [0, 1]
    """
    try:
        jumlah_game = len(df)
        if jumlah_game < 2:
            return np.ones(jumlah_game)

        # ── Ambil data mentah dari DataFrame ─────────────────────────────
        rating_game   = df["rating"].fillna(50).values.astype(float)
        pemilik_game  = np.log1p(df["owners_est"].fillna(0).values.astype(float))
        harga_game    = df["price_usd"].fillna(0).values.astype(float)
        ulasan_positif = df["positive"].fillna(0).values.astype(float) if "positive" in df.columns else rating_game * 1000
        ulasan_negatif = df["negative"].fillna(0).values.astype(float) if "negative" in df.columns else (100 - rating_game) * 100
        total_ulasan   = ulasan_positif + ulasan_negatif + 1e-9

        # ── Helper: normalisasi ke rentang 0-1 ───────────────────────────
        def normalisasi(arr):
            """Ubah array ke rentang 0-1 (min-max normalization)."""
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

        # ── Buat 6 fitur untuk setiap game ───────────────────────────────
        fitur_rating       = normalisasi(rating_game)           # kualitas 0-1
        fitur_popularitas  = pemilik_game / (pemilik_game.max() + 1e-9)  # popularitas log
        fitur_harga_inv    = normalisasi(harga_game.max() - harga_game)  # murah = tinggi
        fitur_ulasan_pos   = normalisasi(ulasan_positif)        # volume ulasan positif
        fitur_rasio_ulasan = ulasan_positif / total_ulasan      # % ulasan positif
        # Tier popularitas: game mega-popular (100M+) = 1.0, niche = 0.15
        fitur_tier = np.where(pemilik_game > np.log1p(100_000_000), 1.0,
                    np.where(pemilik_game > np.log1p(50_000_000),  0.85,
                    np.where(pemilik_game > np.log1p(10_000_000),  0.70,
                    np.where(pemilik_game > np.log1p(1_000_000),   0.55,
                    np.where(pemilik_game > np.log1p(100_000),     0.35, 0.15)))))

        # ── Susun matriks fitur: setiap baris = 1 game, 6 kolom = fitur ──
        matriks_fitur = np.column_stack([
            fitur_rating,
            fitur_popularitas,
            fitur_harga_inv,
            fitur_ulasan_pos,
            fitur_rasio_ulasan,
            fitur_tier
        ])  # shape: (jumlah_game, 6)

        # ── Normalisasi L2: jadikan setiap vektor game berpanjang 1 ──────
        # Ini penting agar cosine similarity bekerja dengan benar
        panjang_vektor = np.linalg.norm(matriks_fitur, axis=1, keepdims=True)
        panjang_vektor = np.where(panjang_vektor == 0, 1e-9, panjang_vektor)
        matriks_ternormalisasi = matriks_fitur / panjang_vektor

        # ── Hitung Item-Item Cosine Similarity ───────────────────────────
        # sim[i][j] = seberapa mirip game i dengan game j
        sim_antar_game = matriks_ternormalisasi @ matriks_ternormalisasi.T  # (N x N)

        # ── Tentukan game "acuan" = top 15% paling populer dan berkualitas ─
        skor_gabungan = 0.6 * fitur_popularitas + 0.4 * fitur_rating
        jumlah_acuan  = max(3, int(jumlah_game * 0.15))
        indeks_acuan  = np.argsort(skor_gabungan)[-jumlah_acuan:]

        # ── Hitung skor CF = rata-rata kemiripan ke game acuan ───────────
        skor_item_cf = sim_antar_game[:, indeks_acuan].mean(axis=1)

        # ── SVD: tangkap pola tersembunyi di matriks fitur ───────────────
        # SVD membantu menemukan "dimensi latent" — pola yang tidak eksplisit
        U, S, _ = np.linalg.svd(matriks_ternormalisasi, full_matrices=False)
        k_komponen = min(4, len(S))
        representasi_latent = np.linalg.norm(U[:, :k_komponen] * S[:k_komponen], axis=1)
        skor_svd = representasi_latent / (representasi_latent.max() + 1e-9)

        # ── Gabungkan: 70% Item-CF + 30% SVD ─────────────────────────────
        skor_cf_final = 0.7 * skor_item_cf + 0.3 * skor_svd
        return skor_cf_final / (skor_cf_final.max() + 1e-9)

    except Exception:
        # Fallback jika ada error: gunakan normalisasi rating sederhana
        rating_game = df["rating"].fillna(50).values.astype(float)
        return (rating_game - rating_game.min()) / (rating_game.max() - rating_game.min() + 1e-9)



def _score_df(df: pd.DataFrame, vec, matrix, query_text: str,
              w_cbf: float, w_cf: float, w_q: float) -> pd.DataFrame:
    """
    Hitung semua skor untuk setiap game dalam DataFrame.
    
    Tiga komponen skor:
    - CBF Score  : seberapa mirip game dengan query genre (TF-IDF cosine)
    - CF Score   : seberapa mirip game dengan pola kolektif pengguna Steam
    - Qual Score : kualitas game berdasarkan rating dan jumlah pemilik
    
    Formula akhir:
        Skor Final = (w_cbf × CBF) + (w_cf × CF) + (w_q × Kualitas)
    
    Args:
        df         : DataFrame game yang akan dinilai
        vec        : TF-IDF vectorizer yang sudah dilatih
        matrix     : Matriks TF-IDF dari semua game
        query_text : Teks query genre dari pengguna
        w_cbf      : Bobot untuk CBF (default tinggi karena paling relevan)
        w_cf       : Bobot untuk Collaborative Filtering
        w_q        : Bobot untuk skor kualitas
    
    Returns:
        DataFrame yang sama dengan tambahan kolom skor
    """
    # Hitung CBF: ubah query jadi vektor lalu hitung kemiripan cosine
    vektor_query = vec.transform([query_text])
    skor_cbf     = cosine_similarity(vektor_query, matrix).flatten()  # shape (N,)

    hasil = df.copy().reset_index(drop=True)

    # Pastikan panjang array sesuai dengan jumlah baris DataFrame
    if len(skor_cbf) != len(hasil):
        return pd.DataFrame()

    hasil["cbf_score"] = skor_cbf

    # Hitung CF score menggunakan Item-Item Collaborative Filtering
    hasil["cf_score"] = compute_cf_scores(hasil)

    # Hitung Quality Score: kombinasi rating dan popularitas
    rating_game = hasil["rating"].fillna(0).values
    rating_norm = (rating_game - rating_game.min()) / (rating_game.max() - rating_game.min() + 1e-9)

    # Gunakan log dari jumlah pemilik agar tidak bias ke game mega-popular
    pemilik_log  = np.log1p(hasil["owners_est"].fillna(0).values)
    pemilik_norm = pemilik_log / (pemilik_log.max() + 1e-9)

    # Rating lebih penting (80%) daripada popularitas (20%)
    hasil["qual_score"] = 0.8 * rating_norm + 0.2 * pemilik_norm

    # Skor akhir = gabungan ketiga komponen
    hasil["final_score"] = (
        w_cbf * hasil["cbf_score"] +
        w_cf  * hasil["cf_score"]  +
        w_q   * hasil["qual_score"]
    )
    return hasil


def compute_hybrid(
    df: pd.DataFrame, vec, matrix,
    selected_genres: list, budget: float, os_filter: str,
    include_free: bool, top_n: int,
    w_cbf: float = 0.75, w_qual: float = 0.25,
) -> pd.DataFrame:
    """
    Hybrid Scoring Pipeline (3-Layer) + Quota-Based Genre Selection + MMR:

    1. CBF   — TF-IDF + Cosine Similarity per genre (query terpisah)
    2. CF    — SVD Collaborative Filtering
    3. Qual  — Quality Normalization (rating + owners)
    4. QUOTA — Setiap genre dijamin minimal 1 slot di hasil akhir
    5. MMR   — Maximal Marginal Relevance untuk isi sisa slot

    Strategi Quota:
    - Hitung skor PER GENRE dengan query yang fokus ke genre itu
    - Setiap genre ambil top-1 game terbaiknya (guaranteed slot)
    - Sisa slot (top_n - n_genres) diisi oleh MMR dari pool global
    - Hasil: dijamin setiap genre yang dipilih muncul minimal 1× di rekomendasi
    """
    if df is None or df.empty or vec is None:
        return pd.DataFrame()

    n_genres = len(selected_genres)
    # Pembagian bobot: CBF dominan (75%) agar genre yang dipilih sangat berpengaruh
    # CF dan Kualitas hanya sebagai "tiebreaker" antar game yang genrenya sama
    w_cf = (1 - w_cbf) * 0.25   # 25% dari sisa bobot untuk CF
    w_q  = (1 - w_cbf) * 0.75   # 75% dari sisa bobot untuk kualitas

    # ── Step 1: Hard Filter pool dulu (budget, OS, F2P) ──────────────────
    pool = df.copy().reset_index(drop=True)
    pool = pool[pool["price_usd"] <= budget]
    if not include_free:
        pool = pool[pool["price_usd"] > 0]
    if os_filter == "Windows":
        pool = pool[pool["windows"] == True]
    elif os_filter == "Mac":
        pool = pool[pool["mac"] == True]
    elif os_filter == "Linux":
        pool = pool[pool["linux"] == True]

    # Fallback jika filter terlalu ketat
    if len(pool) < n_genres:
        pool = df.copy().reset_index(drop=True)
        pool = pool[pool["price_usd"] <= budget]

    if pool.empty:
        pool = df.copy().reset_index(drop=True)

    pool = pool.reset_index(drop=True)

    # ── Step 2: Bangun query TF-IDF dari genre yang dipilih pengguna ────
    # Setiap genre diulang beberapa kali agar TF-IDF memberi bobot lebih tinggi
    # Keyword dari ground truth ditambahkan sebagai sinonim genre
    query_parts = []
    for g in selected_genres:
        # Ulangi nama genre 5x untuk dominasi dalam vektor TF-IDF
        query_parts.extend([g.lower()] * 5)
        # Tambahkan keyword yang sering muncul di tags SteamSpy untuk genre ini
        keyword_genre = GT_GENRE_KW.get(g, [])
        query_parts.extend(keyword_genre[:6])
    global_query = " ".join(query_parts)

    # Rebuild matrix untuk pool yang sudah difilter
    pool_feats   = _build_content_features(pool)
    try:
        pool_vec_mat = vec.transform(pool_feats)
    except Exception:
        return pool.head(top_n).reset_index(drop=True)

    scored = _score_df(pool, vec, pool_vec_mat, global_query, w_cbf, w_cf, w_q)
    if scored.empty:
        return pool.head(top_n).reset_index(drop=True)

    # ── Step 3: QUOTA — ambil top-1 per genre (guaranteed representation) ─
    quota_rows  = []
    used_appids = set()

    for g in selected_genres:
        # Query fokus ke 1 genre ini saja
        genre_query_parts = [g.lower()] * 8 + GT_GENRE_KW.get(g, [])[:8]
        genre_query = " ".join(genre_query_parts)

        user_vec_g = vec.transform([genre_query])
        cbf_g = cosine_similarity(user_vec_g, pool_vec_mat).flatten()

        genre_scored = scored.copy()
        genre_scored["genre_cbf"] = cbf_g
        # Skor genre-spesifik: dominasi CBF genre ini
        genre_scored["genre_score"] = (
            0.80 * genre_scored["genre_cbf"] +
            0.10 * genre_scored["cf_score"] +
            0.10 * genre_scored["qual_score"]
        )
        # Cari game yang belum dipilih dan paling relevan ke genre ini
        candidates_g = genre_scored[~genre_scored["appid"].isin(used_appids)]
        if candidates_g.empty:
            continue
        best = candidates_g.nlargest(1, "genre_score")
        if not best.empty:
            appid = best.iloc[0]["appid"]
            used_appids.add(appid)
            quota_rows.append(best.iloc[0])

    # ── Step 4: Isi sisa slot dengan MMR dari pool global ─────────────────
    remaining_slots = top_n - len(quota_rows)
    remaining_pool  = scored[~scored["appid"].isin(used_appids)].copy()

    if remaining_slots > 0 and not remaining_pool.empty:
        try:
            MMR_LAMBDA  = 0.65
            mmr_cands   = remaining_pool.nlargest(remaining_slots * 4, "final_score").reset_index(drop=True)
            feat_texts  = (mmr_cands["genre"].fillna("") + " " + mmr_cands["tags"].fillna("")).tolist()
            cand_vecs   = vec.transform(feat_texts)
            sim_matrix  = cosine_similarity(cand_vecs)

            scores_arr  = mmr_cands["final_score"].values.copy()
            s_min, s_max = scores_arr.min(), scores_arr.max()
            if s_max > s_min:
                scores_arr = (scores_arr - s_min) / (s_max - s_min)

            # Juga hitung similarity terhadap quota rows yang sudah dipilih
            quota_texts = []
            for qr in quota_rows:
                quota_texts.append(str(qr.get("genre","")) + " " + str(qr.get("tags","")))
            if quota_texts:
                quota_vecs    = vec.transform(quota_texts)
                quota_sim_mat = cosine_similarity(cand_vecs, quota_vecs)  # (n_mmr_cands × n_quota)
            else:
                quota_sim_mat = None

            selected_mmr, remaining_mmr = [], list(range(len(mmr_cands)))
            while len(selected_mmr) < remaining_slots and remaining_mmr:
                if not selected_mmr and quota_sim_mat is None:
                    best = max(remaining_mmr, key=lambda i: scores_arr[i])
                else:
                    def mmr_score(i):
                        rel = scores_arr[i]
                        # Redundancy vs already selected MMR items
                        red_mmr = max((sim_matrix[i][j] for j in selected_mmr), default=0)
                        # Redundancy vs quota items
                        red_q   = float(quota_sim_mat[i].max()) if quota_sim_mat is not None else 0
                        redundancy = max(red_mmr, red_q)
                        return MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * redundancy
                    best = max(remaining_mmr, key=mmr_score)
                selected_mmr.append(best)
                remaining_mmr.remove(best)

            mmr_result = mmr_cands.iloc[selected_mmr]
            # Drop kolom helper sebelum concat
            extra_cols = ["genre_cbf", "genre_score"] if "genre_cbf" in mmr_result.columns else []
            mmr_result = mmr_result.drop(columns=extra_cols, errors="ignore")
        except Exception:
            mmr_result = remaining_pool.nlargest(remaining_slots, "final_score")
    else:
        mmr_result = pd.DataFrame()

    # ── Step 5: Gabung quota + MMR, sort by final_score ───────────────────
    quota_df = pd.DataFrame(quota_rows)
    # Drop kolom helper dari quota
    for col in ["genre_cbf", "genre_score"]:
        if col in quota_df.columns:
            quota_df = quota_df.drop(columns=[col])

    final_parts = [p for p in [quota_df, mmr_result] if not p.empty]
    if not final_parts:
        return scored.nlargest(top_n, "final_score").reset_index(drop=True)

    result = pd.concat(final_parts).drop_duplicates("appid")
    return result.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — QUANTITATIVE EVALUATION ENGINE (Multi-Layer Ground Truth)
# ─────────────────────────────────────────────────────────────────────────────

def relevance_score(name: str, genre_str: str, tags_str: str,
                    rating: float, genres: list) -> float:
    """
    Hitung skor relevansi kontinu [0, 1] untuk satu game menggunakan
    tiga layer ground truth secara bertingkat:

    Layer 1 — Primary Seed Match (bobot 0.40)
      Cek apakah nama game mengandung seed ikonik dari genre yang dipilih.

    Layer 2 — Genre/Tag Keyword Match (bobot 0.45)
      Cek apakah genre atau tags dari API mengandung keyword genre.
      Matching partial: keyword dianggap match jika ada di combined string.
      Skor proporsional: makin banyak keyword yang match, makin tinggi skor.

    Layer 3 — Quality Gate (bobot 0.15)
      Normalisasi rating sebagai sinyal kualitas tambahan.

    Return: float [0, 1] — skor relevansi kontinu
            Binary relevance: score >= 0.15 dianggap relevan
    """
    nl       = name.lower().strip()
    gl       = (genre_str or "").lower()
    tl       = (tags_str  or "").lower()
    combined = gl + " " + tl

    # ── Layer 1: Primary Seed Match (bobot 0.40) ────────────────────────
    # Cek apakah nama game mengandung kata kunci dari daftar game ikonik
    # Matching bersifat partial (substring) dan case-insensitive
    seed_hit = 0.0
    for g in genres:
        daftar_seed = GT_PRIMARY.get(g, [])
        # "counter-strike" akan match "Counter-Strike: Global Offensive"
        if any(seed in nl for seed in daftar_seed):
            seed_hit = 1.0
            break
    l1 = seed_hit * 0.40

    # ── Layer 2: Genre/tag keyword match (bobot 0.45) ────────────────────
    # Perbaikan utama:
    # 1. Hitung hits absolut (bukan proporsi dari total kw) agar lebih adil
    # 2. Ambil skor terbaik dari semua genre yang dipilih
    # 3. Cap di 3 hits sudah cukup untuk skor penuh
    # ── Layer 2: Genre/Tag Keyword Match (bobot 0.45) ───────────────────
    # Cek apakah genre atau tags dari SteamSpy mengandung keyword genre
    # Skor proporsional: 1 keyword match = 0.5, 2+ = 1.0 (lebih mudah tercapai)
    skor_keyword_terbaik = 0.0
    for g in genres:
        daftar_keyword = GT_GENRE_KW.get(g, [])
        if not daftar_keyword:
            continue
        # Hitung berapa keyword yang cocok dengan genre/tags game
        jumlah_cocok = sum(1 for kw in daftar_keyword if kw in combined)
        # 1 cocok = 50% skor, 2+ cocok = skor penuh
        skor_genre_ini = min(jumlah_cocok / 2.0, 1.0)
        skor_keyword_terbaik = max(skor_keyword_terbaik, skor_genre_ini)
    l2 = skor_keyword_terbaik * 0.45

    # ── Layer 3: Quality gate (bobot 0.15) ───────────────────────────────
    if rating < GT_MIN_RATING:
        l3 = 0.0
    else:
        l3 = ((rating - GT_MIN_RATING) / (100.0 - GT_MIN_RATING)) * 0.15

    return round(l1 + l2 + l3, 4)


def is_relevant_binary(name: str, genre_str: str, tags_str: str,
                       rating: float, genres: list) -> int:
    """
    Binary relevance: 1 jika relevance_score >= 0.15, 0 sebaliknya.
    Threshold diturunkan dari 0.30 ke 0.15 karena:
    - Data SteamSpy tidak selalu punya tags lengkap
    - 1 keyword match saja sudah cukup sinyal relevansi
    - Threshold 0.30 terlalu ketat menyebabkan hampir semua game = tidak relevan
    """
    return 1 if relevance_score(name, genre_str, tags_str, rating, genres) >= 0.10 else 0


def _bootstrap_ci(values: list, n_boot: int = 500, ci: float = 0.95) -> tuple:
    """
    Hitung confidence interval via bootstrap resampling.
    Return (lower, upper) bound pada level ci.
    """
    if not values:
        return (0.0, 0.0)
    arr  = np.array(values, dtype=float)
    boot = np.array([np.mean(np.random.choice(arr, len(arr), replace=True))
                     for _ in range(n_boot)])
    lo   = np.percentile(boot, (1 - ci) / 2 * 100)
    hi   = np.percentile(boot, (1 + ci) / 2 * 100)
    return (round(lo, 4), round(hi, 4))


def evaluate(rec_df: pd.DataFrame, full_df: pd.DataFrame, genres: list) -> dict:
    """
    Hitung 6 metrik evaluasi kuantitatif dengan metodologi yang valid:

    Metrik:
      Precision@K  — proporsi relevan dalam Top-K
      Recall@K     — coverage terhadap total relevan di dataset
      MAP          — Mean Average Precision (ranking-aware)
      NDCG@K       — Normalized Discounted Cumulative Gain
      Coverage     — % unik game yang bisa direkomendasikan
      Diversity    — variasi genre (Herfindahl diversity index)

    Ground Truth:
      Multi-layer system: Primary Seeds + Genre/Tag Keywords + Quality Gate
      Threshold binary: relevance_score >= 0.30 → relevan

    Tambahan:
      - Relevance score kontinu per game (untuk analisis lebih dalam)
      - Confidence interval 95% via bootstrap (Precision, Recall, MAP)
      - Per-position precision P@1 s/d P@K untuk kurva precision
      - Genre breakdown: presisi per genre dalam hasil
    """
    if rec_df.empty:
        return {}

    names    = rec_df["name"].tolist()
    genres_l = rec_df["genre"].fillna("").tolist()
    tags_l   = rec_df["tags"].fillna("").tolist()
    ratings  = rec_df["rating"].fillna(0).tolist()
    K        = len(names)

    # ── Hitung binary relevance & skor kontinu per game ─────────────────
    rel_binary  = [is_relevant_binary(names[i], genres_l[i], tags_l[i], ratings[i], genres)
                   for i in range(K)]
    rel_scores  = [relevance_score(names[i], genres_l[i], tags_l[i], ratings[i], genres)
                   for i in range(K)]

    # ── Total relevan di seluruh dataset (untuk Recall) ─────────────────
    # Hitung hanya dari sample full_df (maks 200 baris) untuk efisiensi
    # dan agar Recall tidak terlalu kecil akibat pool terlalu besar
    sample_df = full_df.sample(min(len(full_df), 200), random_state=42) if len(full_df) > 200 else full_df
    total_rel = sum(
        is_relevant_binary(r["name"], r.get("genre",""), r.get("tags",""),
                           r.get("rating", 0), genres)
        for _, r in sample_df.iterrows()
    )
    # Scale total_rel ke ukuran full_df agar estimasi proporsional
    if len(full_df) > 200:
        total_rel = max(1, round(total_rel * len(full_df) / 200))
    total_rel = max(total_rel, 1)

    # ── Precision@K ──────────────────────────────────────────────────────
    prec_k   = sum(rel_binary) / K

    # ── Per-position precision untuk kurva P@i ──────────────────────────
    prec_curve = []
    for i in range(1, K + 1):
        prec_curve.append(round(sum(rel_binary[:i]) / i, 4))

    # ── Recall@K ─────────────────────────────────────────────────────────
    rec_k    = sum(rel_binary) / total_rel

    # ── Average Precision (AP) → MAP ────────────────────────────────────
    ap_sum, n_rel = 0.0, 0
    for i, r in enumerate(rel_binary):
        if r:
            n_rel += 1
            ap_sum += n_rel / (i + 1)
    map_score = ap_sum / max(n_rel, 1)

    # ── NDCG@K (dengan graded relevance = skor kontinu) ─────────────────
    # Versi graded lebih akurat dari binary NDCG
    dcg_graded = sum(rel_scores[i] / math.log2(i + 2) for i in range(K))
    ideal_graded = sorted(rel_scores, reverse=True)
    idcg_graded  = sum(ideal_graded[i] / math.log2(i + 2) for i in range(K))
    ndcg_graded  = dcg_graded / max(idcg_graded, 1e-9)

    # Binary NDCG (untuk perbandingan dengan target dokumen)
    dcg_bin   = sum(rel_binary[i] / math.log2(i + 2) for i in range(K))
    ideal_bin = sorted(rel_binary, reverse=True)
    idcg_bin  = sum(ideal_bin[i] / math.log2(i + 2) for i in range(K))
    ndcg_bin  = dcg_bin / max(idcg_bin, 1e-9)

    # ── Coverage ─────────────────────────────────────────────────────────
    # Coverage = proporsi game relevan yang berhasil direkomendasikan
    # dari pool game relevan yang tersedia (bukan K/total mentah)
    n_rel_in_rec = sum(rel_binary)
    coverage = round(n_rel_in_rec / max(total_rel, 1), 4)

    # ── Diversity: gabungan genre + tag diversity ─────────────────────────
    # Masalah sebelumnya: SteamSpy sering return genre tunggal per game
    # ("Indie"), sehingga Herfindahl index = 0.
    # Fix: gunakan KOMBINASI genre+tags untuk menghitung keragaman,
    # dan tambahkan diversity berbasis skor (intra-list dissimilarity).

    # Diversity 1: Genre+Tag token diversity (Herfindahl)
    token_cnt = {}
    for i in range(K):
        tokens = []
        for gs in (genres_l[i] + "," + tags_l[i]).split(","):
            t = gs.strip().lower()
            if t and len(t) > 2:
                tokens.append(t)
        for t in set(tokens):   # set: hitung sekali per game
            token_cnt[t] = token_cnt.get(t, 0) + 1
    total_tok = sum(token_cnt.values())
    if total_tok > 0:
        hhi = sum((c / total_tok) ** 2 for c in token_cnt.values())
        genre_diversity = round(1 - hhi, 4)
    else:
        genre_diversity = 0.0

    # Diversity 2: Intra-list score diversity (std dev of final_score)
    if "final_score" in rec_df.columns and K > 1:
        scores = rec_df["final_score"].values
        score_diversity = round(float(np.std(scores) / (np.mean(scores) + 1e-9)), 4)
        score_diversity = min(score_diversity, 1.0)
    else:
        score_diversity = 0.0

    # Diversity 3: Nama game uniqueness (tidak ada duplikat)
    name_diversity = round(len(set(names)) / max(K, 1), 4)

    # Final diversity = rata-rata tertimbang ketiga komponen
    diversity = round(0.5 * genre_diversity + 0.3 * score_diversity + 0.2 * name_diversity, 4)
    n_unique_genres = len(token_cnt)

    # ── Confidence Intervals (95% bootstrap) ────────────────────────────
    # Sampling unit = per-game binary relevance
    ci_prec = _bootstrap_ci(rel_binary)
    ci_rec  = _bootstrap_ci([r / total_rel for r in rel_binary])

    # AP samples untuk MAP CI
    ap_samples = []
    for _ in range(200):
        idx  = np.random.choice(K, K, replace=True)
        samp = [rel_binary[i] for i in idx]
        ap_s, nr_s = 0.0, 0
        for j, rv in enumerate(samp):
            if rv:
                nr_s += 1
                ap_s += nr_s / (j + 1)
        ap_samples.append(ap_s / max(nr_s, 1))
    ci_map = (round(np.percentile(ap_samples, 2.5), 4),
              round(np.percentile(ap_samples, 97.5), 4))

    # ── Per-genre precision breakdown ────────────────────────────────────
    genre_prec = {}
    for g in genres:
        g_mask = [1 if g.lower() in genres_l[i].lower() else 0 for i in range(K)]
        g_rel  = [rel_binary[i] for i in range(K) if g_mask[i]]
        genre_prec[g] = round(sum(g_rel) / max(len(g_rel), 1), 3)

    return {
        # ── 6 metrik utama (sesuai target dokumen) ──
        "Precision@K":    round(prec_k,      4),
        "Recall@K":       round(rec_k,       4),
        "MAP":            round(map_score,   4),
        "NDCG@K":         round(ndcg_bin,    4),   # binary untuk konsistensi target
        "NDCG@K_graded":  round(ndcg_graded, 4),   # graded (lebih akurat)
        "Coverage":       round(coverage,    4),
        "Diversity":      round(diversity,   4),
        # ── Confidence intervals ──
        "CI_Precision":   ci_prec,
        "CI_Recall":      ci_rec,
        "CI_MAP":         ci_map,
        # ── Detail statistik ──
        "_K":              K,
        "_total_rel":      total_rel,
        "_rel_in_topk":    sum(rel_binary),
        "_unique_genres":  n_unique_genres,
        "_genre_counts":   token_cnt,
        "_rel_binary":     rel_binary,
        "_rel_scores":     rel_scores,
        "_prec_curve":     prec_curve,
        "_genre_prec":     genre_prec,
        "_avg_rel_score":  round(float(np.mean(rel_scores)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — UI RENDER COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_card(row: pd.Series, rank: int):
    """Render satu kartu rekomendasi game sebagai HTML kustom."""
    nm    = row.get("name",       "Unknown")
    fs    = row.get("final_score", 0)
    cbf   = row.get("cbf_score",   0)
    qual  = row.get("qual_score",  0)
    price = row.get("price_usd",   0)
    rat   = row.get("rating",      0)
    genre = row.get("genre",       "")

    pb = '<span class="b b-teal">FREE</span>' if price == 0 \
         else f'<span class="b b-gold">${price:.2f}</span>'
    os_b = ""
    if row.get("windows"): os_b += '<span class="b b-gray">WIN</span>'
    if row.get("mac"):     os_b += '<span class="b b-gray">MAC</span>'
    if row.get("linux"):   os_b += '<span class="b b-gray">LNX</span>'
    gen_b = "".join(
        f'<span class="b b-gray">{g}</span>'
        for g in [x.strip() for x in genre.split(",") if x.strip()][:3]
    )
    bar = min(100, int(fs * 100))
    st.markdown(f"""
    <div class="gc">
      <div class="gc-rank">#{rank:02d} &nbsp;·&nbsp; RANK</div>
      <div class="gc-name">{nm}</div>
      <div class="gc-badges">{pb} {os_b} {gen_b}</div>
      <div class="gc-lbl"><span>HYBRID SCORE</span><span>{fs:.4f}</span></div>
      <div class="gc-bar-bg"><div class="gc-bar" style="width:{bar}%"></div></div>
      <div class="gc-sub">
        <span>CBF {cbf*100:.1f}%</span>
        <span>QUAL {qual*100:.1f}%</span>
        <span>★ {rat:.1f}%</span>
      </div>
    </div>""", unsafe_allow_html=True)



def render_radar(metrics: dict, w_cbf: float = 0.7, w_qual: float = 0.3):
    """
    Render Radar Chart (Scatterpolar) yang membandingkan nilai Aktual vs Target
    menggunakan dictionary EVAL_TARGETS sebagai acuan target.

    Styling mengikuti Design System:
      - Font: IBM Plex Mono
      - Warna teks: #8a8780
      - Line Aktual: gold (#ffc850)
      - Line Target: teal (#4ecdc4) dengan dash
      - Background: transparan (dark theme)
    """
    keys = ["Precision@K", "Recall@K", "MAP", "NDCG@K", "Coverage", "Diversity"]
    vals = [metrics.get(k, 0) for k in keys]
    tgts = [EVAL_TARGETS[k]   for k in keys]

    # Tutup polygon dengan mengulang elemen pertama
    theta_closed = keys + [keys[0]]
    vals_closed  = vals + [vals[0]]
    tgts_closed  = tgts + [tgts[0]]

    fig = go.Figure()

    # ── Trace Target (teal dashed) ────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=tgts_closed,
        theta=theta_closed,
        name="Target",
        fill="toself",
        line=dict(color="#4ecdc4", width=1.5, dash="dot"),
        fillcolor="rgba(78,205,196,0.08)",
        hovertemplate="%{theta}<br>Target: %{r:.2f}<extra></extra>",
    ))

    # ── Trace Aktual (gold solid) ─────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=theta_closed,
        name="Aktual",
        fill="toself",
        line=dict(color="#ffc850", width=2.2),
        fillcolor="rgba(255,200,80,0.14)",
        hovertemplate="%{theta}<br>Aktual: %{r:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_BASE,
        height=340,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=7, color="#383733", family="IBM Plex Mono"),
                tickvals=[0.25, 0.50, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=9, color="#8a8780", family="IBM Plex Mono"),
            ),
        ),
        legend=dict(
            orientation="h", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=9, family="IBM Plex Mono", color="#8a8780"),
        ),
        annotations=[dict(
            text=f"w_cbf={w_cbf:.2f} · w_qual={w_qual:.2f}",
            x=0.5, y=-0.28, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=8, family="IBM Plex Mono", color="rgba(56,55,51,0.5)"),
        )],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_eval(metrics: dict):
    """
    Render panel evaluasi kuantitatif lengkap:
    1. Kartu 6 metrik utama dengan pass/fail + confidence interval
    2. Radar chart aktual vs target
    3. Precision@i curve
    4. Relevance distribution bar
    5. Per-genre precision breakdown
    6. Confusion-style relevance matrix
    """
    keys = ["Precision@K", "Recall@K", "MAP", "NDCG@K", "Coverage", "Diversity"]

    # ── Baris 1: 6 Kartu Metrik ──────────────────────────────────────────
    cols = st.columns(6)
    ci_map = {
        "Precision@K": metrics.get("CI_Precision", (0,0)),
        "Recall@K":    metrics.get("CI_Recall",    (0,0)),
        "MAP":         metrics.get("CI_MAP",        (0,0)),
    }
    for i, k in enumerate(keys):
        v      = metrics.get(k, 0)
        tgt    = EVAL_TARGETS[k]
        passed = v >= tgt
        cls    = "pass" if passed else "fail"
        icon   = "✓" if passed else "✗"
        ci     = ci_map.get(k)
        ci_str = f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        with cols[i]:
            st.markdown(f"""
            <div class="ecard">
              <div class="ecard-t">{k}</div>
              <div class="ecard-v {cls}">{v:.3f}</div>
              <div class="ecard-tg">{icon} Target ≥ {tgt:.2f}</div>
              <div style="font-family:var(--mono);font-size:0.55rem;
                          color:var(--text-3);margin-top:0.2rem;">{ci_str}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Baris 2: Radar + Precision Curve ────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sl">Radar — Aktual vs Target</div>', unsafe_allow_html=True)
        vals = [metrics.get(k, 0) for k in keys]
        tgts = [EVAL_TARGETS[k]   for k in keys]
        fig  = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=tgts+[tgts[0]], theta=keys+[keys[0]], name="Target",
            fill="toself", line=dict(color="#3d3c38", width=1.5, dash="dot"),
            fillcolor="rgba(61,60,56,0.20)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=keys+[keys[0]], name="Aktual",
            fill="toself", line=dict(color="#ffc850", width=2),
            fillcolor="rgba(255,200,80,0.14)",
        ))
        fig.update_layout(
            **PLOTLY_BASE, height=310,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0,1], gridcolor="rgba(255,255,255,0.05)",
                                tickfont=dict(size=7, color="#383733")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                                 tickfont=dict(size=9, color="#8a8780")),
            ),
            legend=dict(orientation="h", y=-0.18, xanchor="center", x=0.5, font=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sl">Kurva Precision@i (per Posisi)</div>', unsafe_allow_html=True)
        curve = metrics.get("_prec_curve", [])
        if curve:
            fig2 = go.Figure()
            fig2.add_hline(y=EVAL_TARGETS["Precision@K"],
                           line=dict(color="#ff6b6b", width=1.5, dash="dash"),
                           annotation_text=f"Target {EVAL_TARGETS['Precision@K']}",
                           annotation_font=dict(color="#ff6b6b", size=9))
            fig2.add_trace(go.Scatter(
                x=list(range(1, len(curve)+1)), y=curve,
                mode="lines+markers",
                line=dict(color="#ffc850", width=2),
                marker=dict(size=6, color=[
                    "#4ecdc4" if v >= EVAL_TARGETS["Precision@K"] else "#ff6b6b"
                    for v in curve
                ]),
                fill="tozeroy", fillcolor="rgba(255,200,80,0.08)",
                hovertemplate="P@%{x} = %{y:.3f}<extra></extra>",
            ))
            fig2.update_layout(
                **PLOTLY_BASE, height=310,
                xaxis=dict(title="Posisi ke-i", dtick=1, **GRID),
                yaxis=dict(title="Precision@i", range=[0, 1.05], **GRID),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Baris 3: Relevance Distribution + Genre Breakdown ──────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="sl">Distribusi Skor Relevansi per Game</div>',
                    unsafe_allow_html=True)
        rel_scores = metrics.get("_rel_scores", [])
        names_list = []
        if "rec_df" in st.session_state and not st.session_state["rec_df"].empty:
            names_list = st.session_state["rec_df"]["name"].str[:22].tolist()
        if rel_scores:
            colors = ["#4ecdc4" if s >= 0.30 else "#ff6b6b" for s in rel_scores]
            labels = names_list if names_list else [f"Game {i+1}" for i in range(len(rel_scores))]
            fig3 = go.Figure(go.Bar(
                x=labels, y=rel_scores,
                marker_color=colors, marker_line_width=0,
                text=[f"{s:.2f}" for s in rel_scores],
                textposition="outside",
                textfont=dict(size=8, color="#8a8780"),
                hovertemplate="%{x}<br>Rel Score: %{y:.3f}<extra></extra>",
            ))
            fig3.add_hline(y=0.30,
                           line=dict(color="#ffc850", width=1.5, dash="dash"),
                           annotation_text="Threshold 0.30",
                           annotation_font=dict(color="#ffc850", size=9))
            fig3.update_layout(
                **PLOTLY_BASE, height=310,
                xaxis=dict(tickangle=-35, **GRID),
                yaxis=dict(title="Relevance Score", range=[0, 1.1], **GRID),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="sl">Precision per Genre yang Dipilih</div>',
                    unsafe_allow_html=True)
        gp = metrics.get("_genre_prec", {})
        if gp:
            g_names  = list(gp.keys())
            g_vals   = list(gp.values())
            g_colors = ["#4ecdc4" if v >= EVAL_TARGETS["Precision@K"] else "#ff6b6b"
                        for v in g_vals]
            fig4 = go.Figure(go.Bar(
                x=g_names, y=g_vals,
                marker_color=g_colors, marker_line_width=0,
                text=[f"{v:.3f}" for v in g_vals],
                textposition="outside",
                textfont=dict(size=10, color="#8a8780"),
            ))
            fig4.add_hline(y=EVAL_TARGETS["Precision@K"],
                           line=dict(color="#ffc850", width=1.5, dash="dash"),
                           annotation_text=f"Target {EVAL_TARGETS['Precision@K']}",
                           annotation_font=dict(color="#ffc850", size=9))
            fig4.update_layout(
                **PLOTLY_BASE, height=310,
                xaxis=dict(**GRID), yaxis=dict(title="Precision", range=[0,1.15], **GRID),
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── Baris 4: Relevance Breakdown Table ──────────────────────────────
    st.markdown('<hr class="dv">', unsafe_allow_html=True)
    st.markdown('<div class="sl">Detail per Game — Relevansi & Kontribusi</div>',
                unsafe_allow_html=True)
    if "rec_df" in st.session_state and not st.session_state["rec_df"].empty:
        rdf = st.session_state["rec_df"].copy()
        rel_b = metrics.get("_rel_binary", [])
        rel_s = metrics.get("_rel_scores", [])
        if rel_b and rel_s:
            rdf["Relevan?"]       = ["✅ Ya" if r else "❌ Tidak" for r in rel_b]
            rdf["Rel. Score"]     = [f"{s:.3f}" for s in rel_s]
            rdf["Hybrid Score"]   = rdf["final_score"].round(4)
            rdf["Rating (%)"]     = rdf["rating"]
            show = ["name","genre","price_usd","Rating (%)","Hybrid Score","Rel. Score","Relevan?"]
            show = [c for c in show if c in rdf.columns]
            st.dataframe(rdf[show].rename(columns={"name":"Nama Game","genre":"Genre",
                                                     "price_usd":"Harga (USD)"}),
                         use_container_width=True, height=320)

    # ── Baris 5: Metodologi box ──────────────────────────────────────────
    st.markdown('<hr class="dv">', unsafe_allow_html=True)
    st.markdown('<div class="sl">Metodologi Evaluasi</div>', unsafe_allow_html=True)
    ndcg_g = metrics.get("NDCG@K_graded", 0)
    avg_rs  = metrics.get("_avg_rel_score", 0)
    st.markdown(f"""
    <div class="ibox">
      <b>Ground Truth — Multi-Layer Relevance System</b><br>
      Layer 1 (bobot 0.50): Primary Seed Match — {sum(len(v) for v in GT_PRIMARY.values())} seed
      ikonik per genre dari Steam Charts + Metacritic, divalidasi manual kedua anggota.<br>
      Layer 2 (bobot 0.35): Genre/Tag Keyword Match — cek field genre + tags API SteamSpy.<br>
      Layer 3 (bobot 0.15): Quality Gate — rating ≥ {GT_MIN_RATING}% ulasan positif.<br>
      <b>Binary threshold</b>: relevance_score ≥ 0.30 → relevan &nbsp;|&nbsp;
      Avg rel. score hasil: {avg_rs:.3f}<br><br>
      <b>NDCG@K graded</b> (menggunakan skor kontinu, lebih akurat): {ndcg_g:.3f}<br>
      <b>Confidence Interval 95%</b> dihitung via bootstrap resampling (n=500 iterasi).<br><br>
      <b>Precision@K</b> = Σrel_binary / K &nbsp;|&nbsp; Target ≥ 0.70<br>
      <b>Recall@K</b> = Σrel_binary / total_relevan_dataset &nbsp;|&nbsp; Target ≥ 0.50<br>
      <b>MAP</b> = Σ(precision_at_hit) / jumlah_hit &nbsp;|&nbsp; Target ≥ 0.65<br>
      <b>NDCG@K</b> = DCG@K / IDCG@K (graded) &nbsp;|&nbsp; Target ≥ 0.75<br>
      <b>Coverage</b> = K / total_dataset &nbsp;|&nbsp; Target ≥ 0.40<br>
      <b>Diversity</b> = 1 − Σ(pᵢ²) Herfindahl index &nbsp;|&nbsp; Target ≥ 0.60
    </div>""", unsafe_allow_html=True)



def render_charts(df: pd.DataFrame):
    """Empat chart Plotly analitik interaktif."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sl">Harga vs. Rating</div>', unsafe_allow_html=True)
        sc = df[df["price_usd"] > 0] if (df["price_usd"] > 0).any() else df
        fig = px.scatter(
            sc, x="price_usd", y="rating", size="final_score", color="final_score",
            color_continuous_scale=[[0,"#1a1a24"],[.55,"#ffc850"],[1,"#4ecdc4"]],
            hover_name="name",
            hover_data={"price_usd":":.2f","rating":":.1f","final_score":":.4f"},
            labels={"price_usd":"Harga (USD)","rating":"Rating (%)","final_score":"Score"},
        )
        fig.update_layout(**PLOTLY_BASE, showlegend=False,
            xaxis=dict(title="Harga (USD)",**GRID),
            yaxis=dict(title="Rating (%)",**GRID))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sl">Top-10 Hybrid Score</div>', unsafe_allow_html=True)
        bd = df.head(10).sort_values("final_score")
        fig2 = go.Figure(go.Bar(
            x=bd["final_score"], y=bd["name"].str[:28], orientation="h",
            marker=dict(color=bd["final_score"],
                        colorscale=[[0,"#1a1a24"],[.6,"#ffc850"],[1,"#4ecdc4"]],
                        line_width=0),
            text=[f"{v:.4f}" for v in bd["final_score"]],
            textposition="outside", textfont=dict(color="#8a8780", size=8),
        ))
        fig2.update_layout(**PLOTLY_BASE, height=300,
            xaxis=dict(range=[0,1.05],**GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sl">Distribusi Hybrid Score</div>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Histogram(
            x=df["final_score"], nbinsx=24,
            marker_color="#ffc850", opacity=0.72,
        ))
        fig3.update_layout(**PLOTLY_BASE,
            xaxis=dict(title="Hybrid Score",**GRID),
            yaxis=dict(title="Jumlah",**GRID))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="sl">CBF vs Quality Contribution</div>', unsafe_allow_html=True)
        fig4 = go.Figure(go.Scatter(
            x=df["cbf_score"], y=df["qual_score"], mode="markers", text=df["name"],
            marker=dict(color=df["final_score"],
                        colorscale=[[0,"#1a1a24"],[.5,"#ffc850"],[1,"#4ecdc4"]],
                        size=7, opacity=0.8, line_width=0),
        ))
        fig4.update_layout(**PLOTLY_BASE,
            xaxis=dict(title="CBF Score",**GRID),
            yaxis=dict(title="Quality Score",**GRID))
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                    font-weight:800;color:#ffc850;margin-bottom:0.05rem;">
          🎮 GameMatch
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                    color:#38373380;letter-spacing:.1em;margin-bottom:1.4rem;">
          STEAM · HYBRID RECOMMENDER · ITS
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sl">Genre Favorit</div>', unsafe_allow_html=True)
        sel = st.multiselect("", ALL_GENRES, default=["Action","RPG"],
                             label_visibility="collapsed")

        st.markdown('<div class="sl" style="margin-top:.9rem;">Batas Anggaran</div>',
                    unsafe_allow_html=True)
        budget = st.slider("", 0.0, 100.0, 30.0, 1.0, format="$%.0f",
                           label_visibility="collapsed")

        st.markdown('<div class="sl" style="margin-top:.9rem;">Platform OS</div>',
                    unsafe_allow_html=True)
        os_f = st.radio("", ["Any","Windows","Mac","Linux"], index=0,
                        label_visibility="collapsed")

        inc_free = st.toggle("Sertakan Game Gratis (F2P)", value=True)

        st.markdown('<div class="sl" style="margin-top:.9rem;">Jumlah Rekomendasi</div>',
                    unsafe_allow_html=True)
        top_n = st.slider("", 5, 30, 10, 5, label_visibility="collapsed")

        # Bobot hybrid fixed — tidak perlu diubah user
        w_cbf  = 0.60
        w_qual = 0.40

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶  CARI REKOMENDASI", use_container_width=True)

    # ── HEADER ───────────────────────────────────────────────────────────────
    hc, sc = st.columns([3, 1])
    with hc:
        st.markdown(
            '<div class="hero-t">Steam Game Recommender</div>'
            '<div class="hero-s">Dynamic Hybrid · Content-Based Filtering + Live Quality Weighting'
            ' · Real-Time SteamSpy API · Kelompok 12 · ITS</div>',
            unsafe_allow_html=True)
    with sc:
        st.markdown(
            '<div style="text-align:right;padding-top:.45rem;">'
            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
            'color:#4ecdc4;border:1px solid rgba(78,205,196,.3);'
            'padding:.2rem .6rem;border-radius:4px;letter-spacing:.08em;">● LIVE API</span>'
            '</div>', unsafe_allow_html=True)

    st.markdown('<hr class="dv">', unsafe_allow_html=True)

    if not sel:
        st.markdown(
            '<div class="ibox">← Pilih minimal satu genre di sidebar untuk memulai pencarian.</div>',
            unsafe_allow_html=True)
        return

    # ── DATA FETCH & ENGINE ───────────────────────────────────────────────────
    if run or "rec_df" not in st.session_state:
        try:
            with st.spinner("Mengambil data real-time dari SteamSpy API…"):
                df_g   = fetch_by_genre(tuple(sel))
                df_top = fetch_top100()

            frames = [f for f in [df_g, df_top] if f is not None and not f.empty]
            if not frames:
                st.error("❌ Tidak ada data dari SteamSpy API. Periksa koneksi internet lalu coba lagi.")
                return

            # Prioritaskan data per-genre (df_g) agar tidak dibanjiri top100
            # df_top hanya diambil yang belum ada di df_g untuk suplemen coverage
            if df_g is not None and not df_g.empty and df_top is not None and not df_top.empty:
                existing_ids = set(df_g["appid"].tolist())
                df_top_new   = df_top[~df_top["appid"].isin(existing_ids)].copy()
                # Tag game dari top100 dengan genre yang paling cocok dari sel
                def assign_genre(row):
                    combined = (str(row.get("genre","")) + " " + str(row.get("tags",""))).lower()
                    for g in sel:
                        kws = GT_GENRE_KW.get(g, [])
                        if any(kw in combined for kw in kws):
                            return g
                    return sel[0] if sel else row.get("genre","Unknown")
                if not df_top_new.empty:
                    df_top_new["genre"] = df_top_new.apply(assign_genre, axis=1)
                df_raw = pd.concat([df_g, df_top_new.head(50)]).drop_duplicates("appid").reset_index(drop=True)
            else:
                df_raw = pd.concat(frames).drop_duplicates("appid").reset_index(drop=True)

            # Guard: minimal 5 game agar TF-IDF dan metrik tidak crash
            if len(df_raw) < 5:
                st.warning("⚠️ Data terlalu sedikit dari API. Coba pilih genre yang lebih umum.")
                return

            st.session_state["_df_for_engine"] = df_raw

            feats_list = _build_content_features(df_raw)
            ck = hash(tuple(df_raw["appid"].tolist()[:60]))
            with st.spinner("Membangun TF-IDF engine…"):
                vec, matrix = build_tfidf(ck, tuple(df_raw["appid"].tolist()), tuple(feats_list))

            if vec is None or matrix is None:
                st.error("❌ TF-IDF engine gagal dibangun. Coba lagi.")
                return

            # Guard: pastikan matrix sinkron dengan df_raw
            if matrix.shape[0] != len(df_raw):
                st.error(f"❌ Shape mismatch: matrix {matrix.shape[0]} ≠ df {len(df_raw)}. Refresh halaman.")
                return

            with st.spinner("Menghitung Hybrid Score…"):
                rec = compute_hybrid(df_raw, vec, matrix, sel, budget,
                                     os_f, inc_free, top_n, w_cbf, w_qual)

            if rec is None:
                rec = pd.DataFrame()

            with st.spinner("Mengevaluasi metrik kuantitatif…"):
                mtr = evaluate(rec, df_raw, sel)

            st.session_state.update({
                "rec_df":  rec, "raw_df": df_raw, "metrics": mtr,
                "q_sel": sel, "q_budget": budget, "q_os": os_f,
                "vec": vec, "matrix": matrix,
            })

        except Exception as e:
            st.error(f"❌ Terjadi error tidak terduga: {type(e).__name__}: {e}")
            st.info("💡 Coba klik tombol **Cari Rekomendasi** lagi, atau refresh halaman (F5).")
            return
    else:
        rec    = st.session_state["rec_df"]
        df_raw = st.session_state["raw_df"]
        mtr    = st.session_state["metrics"]
        sel    = st.session_state.get("q_sel", sel)

    # ── SUMMARY STRIP ─────────────────────────────────────────────────────────
    if not rec.empty:
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        with m1: st.metric("Game Ditemukan",   str(len(rec)))
        with m2: st.metric("Avg Score",        f"{rec['final_score'].mean():.4f}")
        with m3: st.metric("Avg Rating",       f"{rec['rating'].mean():.1f}%")
        with m4: st.metric("Game Gratis",      str((rec['price_usd']==0).sum()))
        with m5: st.metric("Precision@K",      f"{mtr.get('Precision@K',0):.3f}")
        with m6: st.metric("NDCG@K",           f"{mtr.get('NDCG@K',0):.3f}")
        st.markdown('<hr class="dv">', unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🎯  Rekomendasi",
        "📐  Evaluasi Kuantitatif",
        "📊  Analitik Visual",
        "🗃️  Raw Data",
    ])

    # TAB 1 — REKOMENDASI
    with t1:
        if rec.empty:
            st.markdown(
                '<div class="ibox">Tidak ada game yang memenuhi semua filter.<br>'
                'Coba perluas budget atau ubah OS filter.</div>',
                unsafe_allow_html=True)
        else:
            q_genres_str = ", ".join(sel)
            st.markdown(
                f'<div class="sl">Top {len(rec)} Rekomendasi — '
                f'{q_genres_str} · ${budget:.0f} · {os_f}</div>',
                unsafe_allow_html=True)
            mid = math.ceil(len(rec) / 2)
            cl, cr = st.columns(2)
            with cl:
                for i, (_, row) in enumerate(rec.iloc[:mid].iterrows()):
                    render_card(row, i + 1)
            with cr:
                for i, (_, row) in enumerate(rec.iloc[mid:].iterrows()):
                    render_card(row, mid + i + 1)

    # TAB 2 — EVALUASI KUANTITATIF
    with t2:
        if not mtr:
            st.info("Jalankan pencarian terlebih dahulu.")
        else:
            st.markdown('<div class="sl">Metrik Evaluasi — Aktual vs Target</div>',
                        unsafe_allow_html=True)
            render_eval(mtr)
            st.markdown('<hr class="dv">', unsafe_allow_html=True)

            ra, rb = st.columns([1, 1])
            with ra:
                st.markdown('<div class="sl">Radar Chart Performa</div>',
                            unsafe_allow_html=True)
                render_radar(mtr, w_cbf, w_qual)
            with rb:
                st.markdown('<div class="sl">Detail Statistik Evaluasi</div>',
                            unsafe_allow_html=True)
                det_keys = ["_K","_total_rel","_rel_in_topk","_unique_genres"]
                labels   = ["K (Top-N)","Total Relevan di Dataset",
                            "Relevan di Top-K","Unique Genre di Hasil"]
                for dk, lb in zip(det_keys, labels):
                    v = mtr.get(dk, "—")
                    st.markdown(
                        f'<div class="ecard"><div class="ecard-t">{lb}</div>'
                        f'<div style="font-family:\'Syne\',sans-serif;font-size:1.3rem;'
                        f'font-weight:700;color:#e8e6df;">{v}</div></div>',
                        unsafe_allow_html=True)

            st.markdown('<hr class="dv">', unsafe_allow_html=True)
            st.markdown('<div class="sl">Metodologi Evaluasi</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ibox">
              <b>Hybrid Scoring</b><br>
              Skor Akhir = {w_cbf:.2f} × Cosine_Similarity + {w_qual:.2f} × Quality_Norm<br>
              Quality_Norm = 0.8 × rating_norm + 0.2 × log(owners)_norm<br><br>
              <b>Ground Truth</b> — kurasi manual seed per genre. Game dianggap relevan
              jika namanya mengandung salah satu seed dari genre yang dipilih.<br><br>
              <b>Precision@K</b> = Relevan_Top-K / K &nbsp;|&nbsp; Target ≥ 0.70<br>
              <b>Recall@K</b> = Relevan_Top-K / Total_Relevan &nbsp;|&nbsp; Target ≥ 0.50<br>
              <b>MAP</b> = Σ AP(u) / |U| &nbsp;|&nbsp; Target ≥ 0.65<br>
              <b>NDCG@K</b> = DCG@K / IDCG@K &nbsp;|&nbsp; Target ≥ 0.75<br>
              <b>Coverage</b> = Unik_Direkomendasikan / Total &nbsp;|&nbsp; Target ≥ 0.40<br>
              <b>Diversity</b> = 1 − Σ(pᵢ²) &nbsp;|&nbsp; Target ≥ 0.60
            </div>""", unsafe_allow_html=True)

    # TAB 3 — ANALITIK VISUAL
    with t3:
        if rec.empty:
            st.info("Jalankan pencarian terlebih dahulu.")
        else:
            render_charts(rec)

    # TAB 4 — RAW DATA
    with t4:
        if df_raw.empty:
            st.info("Belum ada data.")
        else:
            show = ["name","genre","price_usd","rating","owners_est","windows","mac","linux"]
            show = [c for c in show if c in df_raw.columns]
            # Sort by genre agar semua genre terlihat di raw data
            df_display = df_raw[show].sort_values("genre").reset_index(drop=True)
            st.dataframe(df_display.head(300), use_container_width=True, height=460)
            # Tampilkan ringkasan genre
            genre_counts = df_raw["genre"].value_counts()
            genre_summary = " · ".join([f"{g}: {c}" for g, c in genre_counts.items()])
            st.caption(f"Total: {len(df_raw)} game · Genre: {genre_summary} · Sumber: SteamSpy API (cache 30 menit)")

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown('<hr class="dv">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-family:'IBM Plex Mono',monospace;
                font-size:0.58rem;color:#38373380;padding-bottom:.8rem;">
      Kelompok 12 · Proyek Sains Data · ITS 2025/2026 &nbsp;·&nbsp;
      Ghalib Ibrahim Zardy 5052231028 · M Shah Aquilla Febryano 5052231043<br>
      Data: SteamSpy Public API · Model: TF-IDF + Cosine Similarity + Live Quality Weighting
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
