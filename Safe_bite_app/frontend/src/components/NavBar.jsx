import React from "react";

export default function NavBar() {
  return (
    <nav style={{
      display: "flex",
      justifyContent: "space-between",
      padding: "12px 20px",
      background: "#0f172a",
      color: "white"
    }}>
      <div style={{ fontWeight: "700" }}>Food Alertness</div>
      <div>v1.0</div>
    </nav>
  );
}
