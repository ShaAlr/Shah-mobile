# 🎮 GameMatch — Steam Game Recommendation System

Sistem rekomendasi game Steam berbasis **Hybrid Recommender** (Content-Based
Filtering + Collaborative Filtering/SVD + Live Quality Weighting), dengan data
real-time dari **SteamSpy Public API**.

Proyek ini terdiri dari dua bagian:

- **Backend** — REST API (FastAPI + scikit-learn) yang menjalankan seluruh
  pipeline: fetch data, TF-IDF, hybrid scoring, dan evaluasi kuantitatif
  (Precision@K, Recall@K, MAP, NDCG@K, Coverage, Diversity). Dikemas dalam
  **Docker** agar bebas konflik library.
- **Frontend** — Antarmuka **React (Vite)** dengan tema *Dark Obsidian + Amber
  Gold*, menampilkan kartu rekomendasi, panel evaluasi, chart analitik, dan
  raw data.

---

## Langkah 1: Kloning Repositori Proyek

Buka terminal (Command Prompt / PowerShell), arahkan ke folder direktori
bersih pilihan Anda, lalu jalankan perintah berikut:

```powershell
git clone <URL_REPOSITORI_ANDA> gamematch
cd gamematch
```

> Struktur folder setelah clone:
>
> ```
> gamematch/
> ├── backend/        # FastAPI + recommendation engine
> ├── frontend/        # React (Vite) UI
> └── docker-compose.yml
> ```

---

## Langkah 2: Menjalankan Backend API via Docker

Kami telah mengemas ekosistem backend ke dalam kontainer Docker berbasis CPU
yang sangat ringan untuk menjamin reprodusibilitas sistem bebas konflik
library. Eksekusi perintah satu baris di bawah ini pada PowerShell Anda
(jalankan dari folder root `gamematch`):

```powershell
docker build -t gamematch-backend ./backend; docker run -d --name gamematch-backend -p 8000:8000 gamematch-backend
```

Atau, jika Anda lebih suka memakai **Docker Compose**:

```powershell
docker compose up -d --build
```

Setelah berhasil, API akan tersedia di:

```
http://localhost:8000
```

Cek apakah backend sudah jalan dengan membuka:

```
http://localhost:8000/api/health
http://localhost:8000/docs        # Swagger UI interaktif
```

> Untuk menghentikan kontainer:
> ```powershell
> docker stop gamematch-backend; docker rm gamematch-backend
> # atau jika pakai docker compose:
> docker compose down
> ```

---

## Langkah 3: Menjalankan Frontend React (Vite)

Buka tab terminal atau PowerShell baru, lalu masuk ke direktori frontend
proyek untuk menginstal dependensi interface serta menyalakan server lokal:

```powershell
cd frontend
npm install
npm run dev
```

Setelah server berjalan, buka browser ke:

```
http://localhost:5173
```

Secara default, frontend akan memanggil backend di `http://localhost:8000`.
Jika backend Anda berjalan di alamat/port lain, salin `.env.example` menjadi
`.env` lalu sesuaikan nilainya:

```powershell
copy .env.example .env
# lalu edit VITE_API_URL di dalam .env
```

---

## Cara Penggunaan

1. Pilih satu atau lebih **genre favorit** di sidebar.
2. Atur **batas anggaran**, **platform OS**, opsi **game gratis (F2P)**, dan
   **jumlah rekomendasi**.
3. Klik **▶ CARI REKOMENDASI**.
4. Jelajahi 4 tab hasil:
   - 🎯 **Rekomendasi** — kartu game dengan Hybrid Score.
   - 📐 **Evaluasi Kuantitatif** — Precision@K, Recall@K, MAP, NDCG@K,
     Coverage, Diversity (vs target), radar chart, dan precision curve.
   - 📊 **Analitik Visual** — scatter Harga vs Rating, Top-10 Hybrid Score,
     distribusi skor, dan kontribusi CBF vs Quality.
   - 🗃️ **Raw Data** — data mentah dari SteamSpy.

---

## Endpoint API (Backend)

| Method | Endpoint          | Deskripsi                                      |
|--------|-------------------|-------------------------------------------------|
| GET    | `/api/health`     | Cek status server                               |
| GET    | `/api/genres`     | Daftar genre + target evaluasi                  |
| POST   | `/api/recommend`  | Hitung rekomendasi hybrid + metrik evaluasi     |

Contoh payload `POST /api/recommend`:

```json
{
  "genres": ["Action", "RPG"],
  "budget": 30,
  "os": "Any",
  "include_free": true,
  "top_n": 10
}
```

---

## Troubleshooting

- **Backend gagal build (error sklearn/numpy)**: pastikan Docker Desktop
  punya cukup memori (≥ 4 GB) dan koneksi internet aktif saat `docker build`.
- **Frontend tidak bisa fetch data / CORS error**: pastikan backend berjalan
  di `http://localhost:8000` dan `VITE_API_URL` di `.env` frontend sudah
  benar.
- **"Data terlalu sedikit dari API"**: SteamSpy kadang membatasi rate
  limit — tunggu beberapa menit lalu coba lagi, atau pilih genre yang lebih
  umum (Action, Indie, RPG).

---

**Kelompok 12 · Proyek Sains Data · ITS 2025/2026**
Ghalib Ibrahim Zardy · M Shah Aquilla Febryano
