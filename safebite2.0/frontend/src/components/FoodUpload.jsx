import React, { useState } from "react";
import axios from "axios";

export default function FoodUpload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5000/scan_food", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Upload failed!");
    }
  };

  return (
    <div className="mb-6 p-4 border rounded shadow">
      <h2 className="text-xl font-bold mb-2">Upload Your Meal</h2>
      <input type="file" accept="image/*" onChange={e => setFile(e.target.files[0])} />
      <button onClick={handleUpload} className="ml-2 p-2 bg-blue-500 text-white rounded">Upload</button>

      {result && (
        <div className="mt-4 p-2 bg-green-50 rounded">
          <p><strong>Food:</strong> {result.food}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
          <p><strong>Calories:</strong> {result.nutrition.calories || "N/A"}</p>
        </div>
      )}
    </div>
  );
}
