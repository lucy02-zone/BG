import React from "react";

function CommentInput({ comment, setComment, analyzeComment, loading }) {
  return (
    <div className="input-box">
      <textarea
        placeholder="Enter a comment to analyze..."
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />

      <button onClick={analyzeComment} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}

export default CommentInput;
