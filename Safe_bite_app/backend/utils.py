# backend/utils.py
'''import os
import json
from typing import Optional

DB_FILE = os.path.join("backend", "db.json")
CAL_DB_FILE = os.path.join("backend", "food_calories.json")

def load_calorie_db():
    """Return list of nutrition dicts."""
    if not os.path.exists(CAL_DB_FILE):
        return []
    with open(CAL_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_nutrition_by_food(food_name: str) -> Optional[dict]:
    caldb = load_calorie_db()
    for item in caldb:
        # case-insensitive match
        if item.get("food", "").strip().lower() == food_name.strip().lower():
            return item
    return None

def append_log(entry: dict):
    logs = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    logs.append(entry)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    return entry
'''
'''
import os
import json
from typing import Optional, Dict

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_FILE = os.path.join(BASE_DIR, "db.json")
CAL_DB_FILE = os.path.join(BASE_DIR, "food_calories.json")


def load_calorie_db() -> list[dict]:
    """Return list of nutrition dicts."""
    if not os.path.exists(CAL_DB_FILE):
        return []
    try:
        with open(CAL_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def get_nutrition_by_food(food_name: str) -> Optional[dict]:
    """Return nutrition info for a given food name (case-insensitive)."""
    caldb = load_calorie_db()
    for item in caldb:
        if item.get("food", "").strip().lower() == food_name.strip().lower():
            return item
    return None


def append_log(entry: Dict) -> Dict:
    """Safely append a log entry to db.json, creating it if missing."""
    logs = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
        except json.JSONDecodeError:
            logs = []

    logs.append(entry)

    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to append log: {e}")

    return entry

'''
import os
import json
from typing import Dict, Optional

def append_log(entry: Dict, db_file: str) -> Dict:
    """Safely append a log entry to the given db_file, creating it if missing."""
    logs = []

    if os.path.exists(db_file):
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
        except json.JSONDecodeError:
            logs = []

    logs.append(entry)

    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to append log: {e}")

    return entry


def get_nutrition_by_food(food_name: str, cal_file: str) -> Optional[dict]:
    """Return nutrition info for a given food from cal_file."""
    if not os.path.exists(cal_file):
        return None
    with open(cal_file, "r", encoding="utf-8") as f:
        caldb = json.load(f)
    for item in caldb:
        if item.get("food", "").strip().lower() == food_name.strip().lower():
            return item
    return None
