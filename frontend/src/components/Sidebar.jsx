import React from "react";

const ALL_GENRES_FALLBACK = [
  "Action", "Adventure", "RPG", "Strategy", "Simulation",
  "Casual", "Indie", "Sports", "Racing", "Horror",
  "Puzzle", "Shooter", "Fighting", "Survival", "Platformer",
];

export default function Sidebar({
  genres, selected, setSelected,
  budget, setBudget,
  osFilter, setOsFilter,
  includeFree, setIncludeFree,
  topN, setTopN,
  onRun, loading,
}) {
  const allGenres = genres && genres.length ? genres : ALL_GENRES_FALLBACK;

  const toggleGenre = (g) => {
    setSelected((prev) =>
      prev.includes(g) ? prev.filter((x) => x !== g) : [...prev, g]
    );
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">🎮 GameMatch</div>
      <div className="sidebar-sub">STEAM · HYBRID RECOMMENDER · ITS</div>

      <div className="sl">Genre Favorit</div>
      <div className="genre-grid">
        {allGenres.map((g) => (
          <div
            key={g}
            className={`genre-chip ${selected.includes(g) ? "active" : ""}`}
            onClick={() => toggleGenre(g)}
          >
            {g}
          </div>
        ))}
      </div>

      <div className="field">
        <div className="sl">Batas Anggaran</div>
        <div className="field-label">${budget.toFixed(0)}</div>
        <input
          type="range" min="0" max="100" step="1"
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
        />
      </div>

      <div className="field">
        <div className="sl">Platform OS</div>
        <div className="radio-row">
          {["Any", "Windows", "Mac", "Linux"].map((o) => (
            <div
              key={o}
              className={`radio-pill ${osFilter === o ? "active" : ""}`}
              onClick={() => setOsFilter(o)}
            >
              {o}
            </div>
          ))}
        </div>
      </div>

      <div
        className="toggle-row"
        onClick={() => setIncludeFree(!includeFree)}
      >
        <div className={`toggle-track ${includeFree ? "on" : ""}`}>
          <div className="toggle-knob" />
        </div>
        <span>Sertakan Game Gratis (F2P)</span>
      </div>

      <div className="field">
        <div className="sl">Jumlah Rekomendasi</div>
        <div className="field-label">{topN}</div>
        <input
          type="range" min="5" max="30" step="5"
          value={topN}
          onChange={(e) => setTopN(Number(e.target.value))}
        />
      </div>

      <button className="btn-run" onClick={onRun} disabled={loading || selected.length === 0}>
        {loading ? "Memuat…" : "▶  CARI REKOMENDASI"}
      </button>
    </aside>
  );
}
