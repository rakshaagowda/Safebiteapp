import React, { useEffect, useState } from "react";
import axios from "axios";

export default function MealLogs() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/logs")
      .then(res => setLogs(res.data.logs))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="mb-6 p-4 border rounded shadow">
      <h2 className="text-xl font-bold mb-2">Meal Logs</h2>
      {logs.length === 0 ? <p>No meals logged yet.</p> :
        <ul>
          {logs.map((log, idx) => (
            <li key={idx} className="mb-2">
              <strong>{log.food}</strong> at {new Date(log.timestamp).toLocaleString()} - Calories: {log.nutrition.calories || "N/A"}
            </li>
          ))}
        </ul>
      }
    </div>
  );
}
