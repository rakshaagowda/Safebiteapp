'''import os
import json
import datetime
from typing import Dict, Any, List, Optional


def get_nutrition_by_food(food_name: str, cal_file: str = "food_calories.json") -> Dict[str, Any]:
    """
    Get nutrition information for a given food name from a JSON file.
    Returns an empty dict if food not found or file missing.
    """
    if not food_name:
        return {}
    if not os.path.exists(cal_file):
        return {}

    try:
        with open(cal_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Use lowercase for case-insensitive matching
        return data.get(food_name.lower(), {})
    except Exception as e:
        print(f"Error reading nutrition file: {e}")
        return {}


def append_log(entry: Dict[str, Any], db_file: str = "db.json"):
    """
    Append a single log entry to a JSON log file.
    Creates the file if it does not exist.
    """
    if not entry or not isinstance(entry, dict):
        return

    logs: List[Dict[str, Any]] = []
    try:
        if os.path.exists(db_file):
            with open(db_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
    except Exception as e:
        print(f"Error reading log file: {e}")
        logs = []

    # Add timestamp if missing
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.datetime.utcnow().isoformat()

    logs.append(entry)

    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error writing log file: {e}")


def read_logs(db_file: str = "db.json") -> List[Dict[str, Any]]:
    """
    Read all logs from the JSON file.
    Returns empty list if file does not exist or cannot be read.
    """
    if not os.path.exists(db_file):
        return []

    try:
        with open(db_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []
'''
import os
import json
import datetime
from typing import Dict, Any, List

# Default calorie/nutrition data file
CAL_FILE = "calorie_data.json"

# Default logs database file
DB_FILE = "db.json"


def get_nutrition_by_food(food_name: str, cal_file: str = CAL_FILE) -> Dict[str, Any]:
    """
    Retrieve nutrition information for a given food from a JSON file.
    Returns an empty dict if the food or file does not exist.
    """
    if not food_name or not os.path.exists(cal_file):
        return {}

    try:
        with open(cal_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(food_name.lower(), {})
    except Exception as e:
        print(f"[ERROR] Reading nutrition file '{cal_file}': {e}")
        return {}


def append_log(entry: Dict[str, Any], db_file: str = DB_FILE):
    """
    Append a log entry to the JSON log file.
    Adds a UTC timestamp if not provided.
    """
    if not entry or not isinstance(entry, dict):
        return

    logs: List[Dict[str, Any]] = []
    if os.path.exists(db_file):
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception as e:
            print(f"[ERROR] Reading log file '{db_file}': {e}")
            logs = []

    if "timestamp" not in entry:
        entry["timestamp"] = datetime.datetime.utcnow().isoformat()

    logs.append(entry)

    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Writing log file '{db_file}': {e}")


def read_logs(db_file: str = DB_FILE) -> List[Dict[str, Any]]:
    """
    Read all meal logs from the JSON file.
    Returns an empty list if the file does not exist or is unreadable.
    """
    if not os.path.exists(db_file):
        return []

    try:
        with open(db_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Reading log file '{db_file}': {e}")
        return []


def get_weekly_risks(risky_foods: List[str]) -> Dict[str, str]:
    """
    Generate weekly advice for foods considered risky.
    Example: high-calorie foods or allergens.
    """
    advice = {}
    for food in risky_foods:
        advice[food] = "Avoid this food this week. Drink plenty of water and eat light meals."
    return advice
