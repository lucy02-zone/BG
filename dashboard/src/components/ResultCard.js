import React from "react";

function ResultCard({ result }) {
  const total = Object.keys(result).length;
  const flagged = Object.values(result).filter(v => v === 1).length;
  const risk = Math.round((flagged / total) * 100);

  return (
    <div className="result-card">
      <h3>Moderation Result</h3>

      <div className="risk-score">
        <span>Risk Score</span>
        <strong>{risk}%</strong>
      </div>

      <div className="grid">
        {Object.entries(result).map(([label, value]) => (
          <div key={label} className={`tag ${value ? "danger" : "safe"}`}>
            {label.toUpperCase()}
            <span>{value ? "FLAGGED" : "SAFE"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}


export default ResultCard;
