import React from "react";
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, BarChart, Bar, ReferenceLine, Cell,
} from "recharts";

const KEYS = ["Precision@K", "Recall@K", "MAP", "NDCG@K", "Coverage", "Diversity"];

export default function EvalPanel({ metrics, targets, wCbf, wQual }) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return <div className="ibox">Jalankan pencarian terlebih dahulu.</div>;
  }

  const ciMap = {
    "Precision@K": metrics.CI_Precision,
    "Recall@K": metrics.CI_Recall,
    "MAP": metrics.CI_MAP,
  };

  const radarData = KEYS.map((k) => ({
    metric: k,
    Aktual: metrics[k] ?? 0,
    Target: targets[k] ?? 0,
  }));

  const precCurve = (metrics._prec_curve || []).map((v, i) => ({
    pos: i + 1,
    value: v,
  }));

  const genrePrec = metrics._genre_prec || {};
  const genrePrecData = Object.entries(genrePrec).map(([g, v]) => ({ genre: g, value: v }));

  const ndcgGraded = metrics["NDCG@K_graded"] ?? 0;
  const avgRelScore = metrics._avg_rel_score ?? 0;

  return (
    <div>
      <div className="sl">Metrik Evaluasi — Aktual vs Target</div>

      {/* 6 metric cards */}
      <div className="eval-grid">
        {KEYS.map((k) => {
          const v = metrics[k] ?? 0;
          const tgt = targets[k] ?? 0;
          const passed = v >= tgt;
          const cls = passed ? "pass" : "fail";
          const icon = passed ? "✓" : "✗";
          const ci = ciMap[k];
          const ciStr = ci ? `95% CI [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : "";
          return (
            <div className="ecard" key={k}>
              <div className="ecard-t">{k}</div>
              <div className={`ecard-v ${cls}`}>{v.toFixed(3)}</div>
              <div className="ecard-tg">{icon} Target ≥ {tgt.toFixed(2)}</div>
              {ciStr && (
                <div style={{
                  fontFamily: "var(--mono)", fontSize: "0.55rem",
                  color: "var(--text-3)", marginTop: "0.2rem",
                }}>{ciStr}</div>
              )}
            </div>
          );
        })}
      </div>

      <br />

      {/* Radar + Precision Curve */}
      <div className="charts-grid">
        <div>
          <div className="sl">Radar — Aktual vs Target</div>
          <ResponsiveContainer width="100%" height={310}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.06)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#8a8780", fontSize: 10, fontFamily: "IBM Plex Mono" }} />
              <PolarRadiusAxis domain={[0, 1]} tick={{ fill: "#383733", fontSize: 8 }} />
              <Radar name="Target" dataKey="Target" stroke="#4ecdc4" fill="#4ecdc4" fillOpacity={0.08} strokeDasharray="4 3" />
              <Radar name="Aktual" dataKey="Aktual" stroke="#ffc850" fill="#ffc850" fillOpacity={0.14} />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: "IBM Plex Mono", color: "#8a8780" }} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="sl">Kurva Precision@i (per Posisi)</div>
          <ResponsiveContainer width="100%" height={310}>
            <LineChart data={precCurve}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="pos" tick={{ fill: "#8a8780", fontSize: 10 }} label={{ value: "Posisi ke-i", position: "insideBottom", fill: "#8a8780", fontSize: 10, dy: 10 }} />
              <YAxis domain={[0, 1.05]} tick={{ fill: "#8a8780", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }} />
              <ReferenceLine y={targets["Precision@K"]} stroke="#ff6b6b" strokeDasharray="4 3"
                label={{ value: `Target ${targets["Precision@K"]}`, fill: "#ff6b6b", fontSize: 9, position: "insideTopRight" }} />
              <Line type="monotone" dataKey="value" stroke="#ffc850" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Genre precision + detail stats */}
      <div className="charts-grid" style={{ marginTop: "1rem" }}>
        <div>
          <div className="sl">Precision per Genre yang Dipilih</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={genrePrecData}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="genre" tick={{ fill: "#8a8780", fontSize: 10 }} />
              <YAxis domain={[0, 1.15]} tick={{ fill: "#8a8780", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }} />
              <ReferenceLine y={targets["Precision@K"]} stroke="#ffc850" strokeDasharray="4 3"
                label={{ value: `Target ${targets["Precision@K"]}`, fill: "#ffc850", fontSize: 9, position: "insideTopRight" }} />
              <Bar dataKey="value">
                {genrePrecData.map((entry, i) => (
                  <Cell key={i} fill={entry.value >= targets["Precision@K"] ? "#4ecdc4" : "#ff6b6b"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="sl">Detail Statistik Evaluasi</div>
          {[
            ["K (Top-N)", metrics._K],
            ["Total Relevan di Dataset", metrics._total_rel],
            ["Relevan di Top-K", metrics._rel_in_topk],
            ["Unique Genre di Hasil", metrics._unique_genres],
          ].map(([label, val]) => (
            <div className="ecard" key={label}>
              <div className="ecard-t">{label}</div>
              <div style={{ fontFamily: "Syne, sans-serif", fontSize: "1.3rem", fontWeight: 700, color: "#e8e6df" }}>
                {val}
              </div>
            </div>
          ))}
        </div>
      </div>

      <hr className="dv" />
      <div className="sl">Metodologi Evaluasi</div>
      <div className="ibox">
        <b>Hybrid Scoring</b><br />
        Skor Akhir = {wCbf.toFixed(2)} × Cosine_Similarity + {wQual.toFixed(2)} × Quality_Norm<br />
        Quality_Norm = 0.8 × rating_norm + 0.2 × log(owners)_norm<br /><br />
        <b>NDCG@K graded</b> (skor kontinu, lebih akurat): {ndcgGraded.toFixed(3)}<br />
        <b>Avg. Relevance Score</b> hasil: {avgRelScore.toFixed(3)}<br /><br />
        <b>Precision@K</b> = Relevan_Top-K / K &nbsp;|&nbsp; Target ≥ {targets["Precision@K"]}<br />
        <b>Recall@K</b> = Relevan_Top-K / Total_Relevan &nbsp;|&nbsp; Target ≥ {targets["Recall@K"]}<br />
        <b>MAP</b> = Σ AP(u) / |U| &nbsp;|&nbsp; Target ≥ {targets["MAP"]}<br />
        <b>NDCG@K</b> = DCG@K / IDCG@K &nbsp;|&nbsp; Target ≥ {targets["NDCG@K"]}<br />
        <b>Coverage</b> = Unik_Direkomendasikan / Total &nbsp;|&nbsp; Target ≥ {targets["Coverage"]}<br />
        <b>Diversity</b> = 1 − Σ(pᵢ²) &nbsp;|&nbsp; Target ≥ {targets["Diversity"]}
      </div>
    </div>
  );
}
