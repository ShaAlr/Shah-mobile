import React from "react";

export default function RawDataTable({ rawData, rawTotal, genreCounts }) {
  if (!rawData || rawData.length === 0) {
    return <div className="ibox">Belum ada data.</div>;
  }

  const columns = ["name", "genre", "price_usd", "rating", "owners_est", "windows", "mac", "linux"];
  const labels = {
    name: "Nama Game", genre: "Genre", price_usd: "Harga (USD)",
    rating: "Rating (%)", owners_est: "Owners (est)",
    windows: "WIN", mac: "MAC", linux: "LNX",
  };
  const cols = columns.filter((c) => c in rawData[0]);

  const genreSummary = Object.entries(genreCounts || {})
    .map(([g, c]) => `${g}: ${c}`)
    .join(" · ");

  return (
    <div>
      <div className="table-wrap">
        <table className="dataframe">
          <thead>
            <tr>
              {cols.map((c) => <th key={c}>{labels[c] || c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rawData.map((row, i) => (
              <tr key={i}>
                {cols.map((c) => {
                  let val = row[c];
                  if (typeof val === "boolean") val = val ? "✓" : "—";
                  if (c === "price_usd" && typeof val === "number") val = `$${val.toFixed(2)}`;
                  if (c === "rating" && typeof val === "number") val = `${val.toFixed(1)}%`;
                  return <td key={c}>{String(val ?? "")}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="caption">
        Total: {rawTotal} game · Genre: {genreSummary} · Sumber: SteamSpy API (cache 30 menit)
      </div>
    </div>
  );
}
