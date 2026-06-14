import React, { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import GameCard from "./components/GameCard.jsx";
import EvalPanel from "./components/EvalPanel.jsx";
import Charts from "./components/Charts.jsx";
import RawDataTable from "./components/RawDataTable.jsx";
import { getGenres, getRecommendations } from "./api.js";

const TABS = [
  { id: "rec", label: "🎯  Rekomendasi" },
  { id: "eval", label: "📐  Evaluasi Kuantitatif" },
  { id: "charts", label: "📊  Analitik Visual" },
  { id: "raw", label: "🗃️  Raw Data" },
];

export default function App() {
  const [genres, setGenres] = useState([]);
  const [evalTargets, setEvalTargets] = useState({});

  const [selected, setSelected] = useState(["Action", "RPG"]);
  const [budget, setBudget] = useState(30);
  const [osFilter, setOsFilter] = useState("Any");
  const [includeFree, setIncludeFree] = useState(true);
  const [topN, setTopN] = useState(10);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("rec");

  useEffect(() => {
    getGenres()
      .then((data) => {
        setGenres(data.genres || []);
        setEvalTargets(data.eval_targets || {});
      })
      .catch(() => {
        // fallback already handled in Sidebar / EvalPanel defaults
      });
  }, []);

  const handleRun = async () => {
    if (selected.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getRecommendations({
        genres: selected,
        budget,
        os: osFilter,
        include_free: includeFree,
        top_n: topN,
      });
      setResult(data);
      if (data.eval_targets) setEvalTargets(data.eval_targets);
    } catch (e) {
      setError(e.message || "Terjadi kesalahan tidak terduga.");
    } finally {
      setLoading(false);
    }
  };

  const rec = result?.recommendations || [];
  const mid = Math.ceil(rec.length / 2);
  const left = rec.slice(0, mid);
  const right = rec.slice(mid);

  return (
    <div className="app-shell">
      <Sidebar
        genres={genres}
        selected={selected}
        setSelected={setSelected}
        budget={budget}
        setBudget={setBudget}
        osFilter={osFilter}
        setOsFilter={setOsFilter}
        includeFree={includeFree}
        setIncludeFree={setIncludeFree}
        topN={topN}
        setTopN={setTopN}
        onRun={handleRun}
        loading={loading}
      />

      <main className="main">
        <div className="header-row">
          <div>
            <div className="hero-t">Steam Game Recommender</div>
            <div className="hero-s">
              Dynamic Hybrid · Content-Based Filtering + Live Quality Weighting
              · Real-Time SteamSpy API · Kelompok 12 · ITS
            </div>
          </div>
          <span className="live-badge">● LIVE API</span>
        </div>

        <hr className="dv" />

        {selected.length === 0 && (
          <div className="ibox">← Pilih minimal satu genre di sidebar untuk memulai pencarian.</div>
        )}

        {loading && <div className="spinner-text">Mengambil data &amp; menghitung rekomendasi…</div>}

        {error && (
          <div className="error-box">
            ❌ {error}
            <br />💡 Coba klik tombol <b>Cari Rekomendasi</b> lagi, atau refresh halaman.
          </div>
        )}

        {result && !loading && (
          <>
            {rec.length > 0 && (
              <>
                <div className="metric-strip">
                  <div className="metric">
                    <div className="metric-label">Game Ditemukan</div>
                    <div className="metric-value">{rec.length}</div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Avg Score</div>
                    <div className="metric-value">
                      {(rec.reduce((s, r) => s + r.final_score, 0) / rec.length).toFixed(4)}
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Avg Rating</div>
                    <div className="metric-value">
                      {(rec.reduce((s, r) => s + r.rating, 0) / rec.length).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Game Gratis</div>
                    <div className="metric-value">
                      {rec.filter((r) => r.price_usd === 0).length}
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Precision@K</div>
                    <div className="metric-value">{(result.metrics?.["Precision@K"] ?? 0).toFixed(3)}</div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">NDCG@K</div>
                    <div className="metric-value">{(result.metrics?.["NDCG@K"] ?? 0).toFixed(3)}</div>
                  </div>
                </div>
                <hr className="dv" />
              </>
            )}

            <div className="tabs">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  className={`tab ${activeTab === t.id ? "active" : ""}`}
                  onClick={() => setActiveTab(t.id)}
                >
                  {t.label}
                </button>
              ))}
            </div>

            <div className="tab-panel">
              {activeTab === "rec" && (
                rec.length === 0 ? (
                  <div className="ibox">
                    Tidak ada game yang memenuhi semua filter.<br />
                    Coba perluas budget atau ubah OS filter.
                  </div>
                ) : (
                  <>
                    <div className="sl">
                      Top {rec.length} Rekomendasi — {result.query.genres.join(", ")} ·
                      &nbsp;${result.query.budget.toFixed(0)} · {result.query.os}
                    </div>
                    <div className="cards-grid">
                      <div>
                        {left.map((row, i) => (
                          <GameCard key={row.appid ?? i} row={row} rank={i + 1} />
                        ))}
                      </div>
                      <div>
                        {right.map((row, i) => (
                          <GameCard key={row.appid ?? i} row={row} rank={mid + i + 1} />
                        ))}
                      </div>
                    </div>
                  </>
                )
              )}

              {activeTab === "eval" && (
                <EvalPanel
                  metrics={result.metrics}
                  targets={evalTargets}
                  wCbf={result.query?.w_cbf ?? 0.6}
                  wQual={result.query?.w_qual ?? 0.4}
                />
              )}

              {activeTab === "charts" && <Charts recommendations={rec} />}

              {activeTab === "raw" && (
                <RawDataTable
                  rawData={result.raw_data}
                  rawTotal={result.raw_total}
                  genreCounts={result.genre_counts}
                />
              )}
            </div>
          </>
        )}

        <hr className="dv" />
        <div className="footer">
          Kelompok 12 · Proyek Sains Data · ITS 2025/2026 &nbsp;·&nbsp;
          Ghalib Ibrahim Zardy 5052231028 · M Shah Aquilla Febryano 5052231043<br />
          Data: SteamSpy Public API · Model: TF-IDF + Cosine Similarity + Live Quality Weighting
        </div>
      </main>
    </div>
  );
}
