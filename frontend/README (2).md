# 🎮 GameMatch — Steam Game Recommendation System

Sistem rekomendasi game Steam berbasis **Hybrid Recommender** (Content-Based
Filtering + Collaborative Filtering/SVD + Live Quality Weighting), dengan data
real-time dari **SteamSpy Public API**.

Repositori: `https://github.com/ShaAlr/Shah-mobile`

Proyek ini bisa dijalankan dengan **dua cara**:

| Cara | Teknologi | Cocok untuk |
|---|---|---|
| **Cara A** | Streamlit (`app.py`) | Demo cepat, tanpa Docker |
| **Cara B** | FastAPI + Docker + React | Deployment lengkap |

---

## Persiapan Awal (Wajib untuk Keduanya)

1. **Python 3.10+** — https://www.python.org/downloads/
   > Saat install, centang **"Add Python to PATH"**
2. **Node.js 18+** *(hanya untuk Cara B)* — https://nodejs.org/
3. **Docker Desktop** *(hanya untuk Cara B)* — https://www.docker.com/products/docker-desktop/

---

## Cara A: Menjalankan via Streamlit

### Langkah 1 — Download Project

1. Buka `https://github.com/ShaAlr/Shah-mobile`
2. Klik **`<> Code`** → **`Download ZIP`**
3. Extract ZIP ke folder pilihan (misal `Downloads`)
4. Buka **PowerShell**, masuk ke folder root project:
   ```powershell
   cd "C:\Users\NAMA-USER\Downloads\Shah-mobile-main\gamematch"
   ```

### Langkah 2 — Setup Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> Jika muncul error *"running scripts is disabled"*, jalankan dulu:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> lalu ulangi `.\venv\Scripts\Activate.ps1`

### Langkah 3 — Install Library

```powershell
pip install -r requirements_streamlit.txt
```

### Langkah 4 — Jalankan Aplikasi

```powershell
streamlit run app.py
```

Browser akan otomatis terbuka ke `http://localhost:8501` — aplikasi
GameMatch langsung tampil.

---

## Cara B: Menjalankan via Docker + React

### Langkah 1 — Download Project

Sama seperti Cara A Langkah 1 di atas.

### Langkah 2 — Rename File Backend

Masuk ke folder `backend`:
```powershell
cd "C:\Users\NAMA-USER\Downloads\Shah-mobile-main\gamematch\backend"
```

Rename file (jika belum):
```powershell
ren app.py main.py
```

### Langkah 3 — Setup Virtual Environment Backend

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Langkah 4 — Jalankan Backend

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Tunggu hingga muncul `Application startup complete`.

Cek di browser: `http://localhost:8000/api/health`
→ Harus muncul `{"status":"ok"}`

### Langkah 5 — Jalankan Frontend React

Buka **jendela PowerShell baru**, lalu:

```powershell
cd "C:\Users\NAMA-USER\Downloads\Shah-mobile-main\gamematch\frontend"
npm install
npm run dev
```

Buka browser ke `http://localhost:5173`

### Langkah 5 (Alternatif) — Jalankan Backend via Docker

Pastikan **Docker Desktop** terbuka dan statusnya **"Engine running"**, lalu:

```powershell
cd "C:\Users\NAMA-USER\Downloads\Shah-mobile-main\gamematch"
docker compose up -d
```

Cek: `http://localhost:8000/api/health` → `{"status":"ok"}`

Lanjut jalankan frontend (Langkah 5 di atas).

---

## Cara Menggunakan Aplikasi

1. Pilih **genre favorit** di sidebar
2. Atur **budget**, **platform OS**, opsi **game gratis (F2P)**, dan
   **jumlah rekomendasi**
3. Klik **▶ CARI REKOMENDASI**
4. Lihat hasil di 4 tab:
   - 🎯 **Rekomendasi**
   - 📐 **Evaluasi Kuantitatif** (Precision@K, Recall@K, MAP, NDCG@K)
   - 📊 **Analitik Visual**
   - 🗃️ **Raw Data**

---

## Struktur Folder

```
Shah-mobile/
├── app.py                      ← Streamlit app (Cara A)
├── requirements_streamlit.txt  ← Library untuk Streamlit
├── backend/
│   ├── main.py                 ← FastAPI backend (Cara B)
│   ├── requirements.txt        ← Library untuk FastAPI
│   └── Dockerfile
├── frontend/                   ← React Vite UI (Cara B)
│   ├── src/
│   ├── package.json
│   └── ...
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Troubleshooting

| Error | Solusi |
|---|---|
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `Could not import module "main"` | Pastikan sudah rename `app.py` → `main.py` di folder `backend` |
| `Cannot find path '\backend'` | Path lengkap: `...\Shah-mobile-main\gamematch\backend` |
| `ModuleNotFoundError: fastapi` | venv belum aktif atau `pip install` belum selesai |
| Rekomendasi lama muncul | SteamSpy API lambat, tunggu max 3 detik lalu otomatis pakai data fallback |

---

**Kelompok 12 · Proyek Sains Data · ITS 2025/2026**
Ghalib Ibrahim Zardy 5052231028 · M Shah Aquilla Febryano 5052231043
