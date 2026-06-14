import React from "react";

export default function GameCard({ row, rank }) {
  const fs = row.final_score ?? 0;
  const cbf = row.cbf_score ?? 0;
  const qual = row.qual_score ?? 0;
  const price = row.price_usd ?? 0;
  const rating = row.rating ?? 0;
  const genre = row.genre ?? "";
  const bar = Math.min(100, Math.round(fs * 100));

  const genreBadges = genre
    .split(",")
    .map((g) => g.trim())
    .filter(Boolean)
    .slice(0, 3);

  return (
    <div className="gc">
      <div className="gc-rank">#{String(rank).padStart(2, "0")} &nbsp;·&nbsp; RANK</div>
      <div className="gc-name">{row.name}</div>
      <div className="gc-badges">
        {price === 0 ? (
          <span className="b b-teal">FREE</span>
        ) : (
          <span className="b b-gold">${price.toFixed(2)}</span>
        )}
        {row.windows && <span className="b b-gray">WIN</span>}
        {row.mac && <span className="b b-gray">MAC</span>}
        {row.linux && <span className="b b-gray">LNX</span>}
        {genreBadges.map((g, i) => (
          <span key={i} className="b b-gray">{g}</span>
        ))}
      </div>
      <div className="gc-lbl">
        <span>HYBRID SCORE</span>
        <span>{fs.toFixed(4)}</span>
      </div>
      <div className="gc-bar-bg">
        <div className="gc-bar" style={{ width: `${bar}%` }} />
      </div>
      <div className="gc-sub">
        <span>CBF {(cbf * 100).toFixed(1)}%</span>
        <span>QUAL {(qual * 100).toFixed(1)}%</span>
        <span>★ {rating.toFixed(1)}%</span>
      </div>
    </div>
  );
}
