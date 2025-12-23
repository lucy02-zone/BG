import React, { useState } from "react";
import axios from "axios";
import CommentInput from "./components/CommentInput";
import ResultCard from "./components/ResultCard";
import "./App.css";

function App() {
  const [comment, setComment] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeComment = async () => {
    if (!comment.trim()) return;

    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/predict", {
        text: comment
      });
      setResult(response.data);
    } catch (error) {
      alert("Backend not running!");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <h1>🛡️ Toxic Comment Moderation</h1>

      <CommentInput
        comment={comment}
        setComment={setComment}
        analyzeComment={analyzeComment}
        loading={loading}
      />

      {result && <ResultCard result={result} />}
    </div>
  );
}

export default App;
