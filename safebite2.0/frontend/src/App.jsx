import React from "react";
import FoodUpload from "./components/FoodUpload";
import MealLogs from "./components/MealLogs";
import WeeklyReport from "./components/WeeklyReport";

export default function App() {
  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">SafeBite Dashboard</h1>
      <FoodUpload />
      <MealLogs />
      <WeeklyReport />
    </div>
  );
}
