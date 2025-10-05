import React, { useState } from "react";

export default function CalorieScanner() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleFile(e) {
    const f = e.target.files[0];
    setFile(f);
    if (f) {
      setPreview(URL.createObjectURL(f));
    } else {
      setPreview(null);
    }
  }

  async function handleUpload() {
    if (!file) return alert("Select an image first");
    setLoading(true);
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/scan_food", {
        method: "POST",
        body: form
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "Scanning failed");
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error(err);
      alert("Server error. Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      maxWidth: 720, margin: "12px auto", padding: 16, borderRadius: 8, boxShadow: "0 2px 10px rgba(0,0,0,0.08)"
    }}>
      <h2>Calorie Scanner</h2>
      <input type="file" accept="image/*" onChange={handleFile} />
      {preview && <div style={{ marginTop: 12 }}>
        <img src={preview} alt="preview" style={{ maxWidth: "100%", height: "auto", borderRadius: 8 }} />
      </div>}
      <div style={{ marginTop: 12 }}>
        <button onClick={handleUpload} disabled={loading} style={{ padding: "8px 12px" }}>
          {loading ? "Scanning..." : "Upload & Scan"}
        </button>
      </div>

      {result && (
        <div style={{ marginTop: 16, background: "#f9fafb", padding: 12, borderRadius: 8 }}>
          <h3>Result</h3>
          <p><strong>Food:</strong> {result.food}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
          {result.nutrition ? (
            <ul>
              <li>Calories: {result.nutrition.calories}</li>
              <li>Carbs: {result.nutrition.carbs} g</li>
              <li>Protein: {result.nutrition.protein} g</li>
              <li>Fat: {result.nutrition.fat} g</li>
            </ul>
          ) : <p>No nutrition info found for this food.</p>}
        </div>
      )}
    </div>
  );
}
