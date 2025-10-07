import React, { useEffect, useState } from "react";
import axios from "axios";

export default function WeeklyReport() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/predict_sick")
      .then(res => setData(res.data))
      .catch(err => console.error(err));
  }, []);

  if (!data) return <p>Loading weekly report...</p>;

  return (
    <div className="mb-6 p-4 border rounded shadow">
      <h2 className="text-xl font-bold mb-2">Weekly Food Risk Report</h2>
      {data.risky_foods.length === 0 ? (
        <p>No risky foods detected this week. 👍</p>
      ) : (
        data.risky_foods.map((food, idx) => (
          <div key={idx} className="mb-3 p-2 bg-red-50 rounded">
            <p><strong>Food:</strong> {food}</p>
            <p><strong>Advice:</strong> {data.advice[food]}</p>
          </div>
        ))
      )}
    </div>
  );
}
