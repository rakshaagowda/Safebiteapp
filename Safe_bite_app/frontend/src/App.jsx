import React from "react";
import NavBar from "./components/NavBar";
import CalorieScanner from "./components/CalorieScanner";

export default function App() {
  return (
    <div>
      <NavBar />
      <main style={{ padding: 20 }}>
        <h1>Food Alertness App</h1>
        <CalorieScanner />
      </main>
    </div>
  );
}
