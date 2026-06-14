const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function getGenres() {
  const res = await fetch(`${API_BASE}/api/genres`);
  if (!res.ok) throw new Error("Gagal memuat daftar genre.");
  return res.json();
}

export async function getRecommendations(payload) {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = "Terjadi kesalahan pada server.";
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}
