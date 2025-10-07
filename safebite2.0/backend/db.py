import sqlite3
from typing import List, Dict, Any

DB_FILE = "backend.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food TEXT,
            calories REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_meal(food: str, calories: float):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO meals (food, calories) VALUES (?, ?)', (food, calories))
    conn.commit()
    conn.close()

def get_meals() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT food, calories, timestamp FROM meals ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return [{"food": r[0], "calories": r[1], "timestamp": r[2]} for r in rows]

def get_weekly_risks(risky_foods: List[str]) -> Dict[str, Any]:
    """Return weekly advice for risky foods"""
    advice = {}
    for food in risky_foods:
        advice[food] = "Avoid this food this week. Drink plenty of water and eat light meals."  # Placeholder advice
    return advice
