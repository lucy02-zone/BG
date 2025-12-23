import React from "react";

function ResultCard({ result }) {
  return (
    <div className="result-card">
      <h3>Moderation Result</h3>

      <div className="grid">
        {Object.entries(result).map(([label, value]) => (
          <div
            key={label}
            className={`tag ${value ? "danger" : "safe"}`}
          >
            {label.toUpperCase()}
            <span>{value ? "FLAGGED" : "SAFE"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ResultCard;
