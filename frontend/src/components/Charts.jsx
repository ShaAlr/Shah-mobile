import React from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell,
} from "recharts";

function buildHistogram(values, bins = 12) {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / bins || 1;
  const counts = Array.from({ length: bins }, () => 0);
  values.forEach((v) => {
    let idx = Math.floor((v - min) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  });
  return counts.map((c, i) => ({
    bucket: (min + i * width).toFixed(2),
    count: c,
  }));
}

export default function Charts({ recommendations }) {
  if (!recommendations || recommendations.length === 0) {
    return <div className="ibox">Jalankan pencarian terlebih dahulu.</div>;
  }

  const priceRatingData = recommendations.map((r) => ({
    name: r.name,
    price: r.price_usd,
    rating: r.rating,
    score: r.final_score,
  }));

  const top10 = [...recommendations]
    .slice(0, 10)
    .sort((a, b) => a.final_score - b.final_score)
    .map((r) => ({ name: r.name.slice(0, 28), score: r.final_score }));

  const histData = buildHistogram(recommendations.map((r) => r.final_score), 12);

  const cbfQualData = recommendations.map((r) => ({
    name: r.name,
    cbf: r.cbf_score,
    qual: r.qual_score,
    score: r.final_score,
  }));

  return (
    <div>
      <div className="charts-grid">
        <div>
          <div className="sl">Harga vs. Rating</div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="price" name="Harga (USD)" tick={{ fill: "#8a8780", fontSize: 10 }}
                label={{ value: "Harga (USD)", position: "insideBottom", fill: "#8a8780", fontSize: 10, dy: 10 }} />
              <YAxis dataKey="rating" name="Rating (%)" tick={{ fill: "#8a8780", fontSize: 10 }}
                label={{ value: "Rating (%)", angle: -90, position: "insideLeft", fill: "#8a8780", fontSize: 10 }} />
              <ZAxis dataKey="score" range={[40, 200]} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }}
                formatter={(v, n) => [typeof v === "number" ? v.toFixed(2) : v, n]} />
              <Scatter data={priceRatingData} fill="#ffc850" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="sl">Top-10 Hybrid Score</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={top10} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" />
              <XAxis type="number" domain={[0, 1.05]} tick={{ fill: "#8a8780", fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={140} tick={{ fill: "#8a8780", fontSize: 9 }} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }}
                formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="score" fill="#ffc850" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="charts-grid" style={{ marginTop: "1rem" }}>
        <div>
          <div className="sl">Distribusi Hybrid Score</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={histData}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="bucket" tick={{ fill: "#8a8780", fontSize: 9 }}
                label={{ value: "Hybrid Score", position: "insideBottom", fill: "#8a8780", fontSize: 10, dy: 10 }} />
              <YAxis tick={{ fill: "#8a8780", fontSize: 10 }}
                label={{ value: "Jumlah", angle: -90, position: "insideLeft", fill: "#8a8780", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }} />
              <Bar dataKey="count" fill="#ffc850" fillOpacity={0.72} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="sl">CBF vs Quality Contribution</div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="cbf" name="CBF Score" tick={{ fill: "#8a8780", fontSize: 10 }}
                label={{ value: "CBF Score", position: "insideBottom", fill: "#8a8780", fontSize: 10, dy: 10 }} />
              <YAxis dataKey="qual" name="Quality Score" tick={{ fill: "#8a8780", fontSize: 10 }}
                label={{ value: "Quality Score", angle: -90, position: "insideLeft", fill: "#8a8780", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#13131a", border: "1px solid var(--border)" }}
                formatter={(v, n) => [typeof v === "number" ? v.toFixed(3) : v, n]} />
              <Scatter data={cbfQualData} fill="#4ecdc4" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
