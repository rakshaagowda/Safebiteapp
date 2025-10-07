'''import os
import sys
import datetime
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add backend folder to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_nutrition_by_food, append_log, read_logs
from models.food_classifier import load_model_and_labels, predict_food_from_path
from models.sickness_predictor import predict_sickness, load_sickness_model_if_any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_FILE = os.path.join(BASE_DIR, "db.json")
CAL_DB_FILE = os.path.join(BASE_DIR, "food_calories.json")
MODEL_FILE = os.path.join(BASE_DIR, "food_classifier.h5")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

def allowed_file(filename: str | None) -> bool:
    if filename is None:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# load models (lazy load)
LABELS, TF_MODEL = load_model_and_labels(MODEL_FILE)  # returns (labels_list, model) or (None,None)
SICKNESS_MODEL = load_sickness_model_if_any(os.path.join(BASE_DIR, "sickness_model.pkl"))

@app.route("/")
def home():
    return "SafeBite backend is running!"

@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        food_name, confidence = predict_food_from_path(save_path, TF_MODEL, LABELS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    nutrition = get_nutrition_by_food(food_name, CAL_DB_FILE) or {}
    log_entry = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": float(confidence),
        "nutrition": nutrition,
        "source_file": unique_name
    }
    append_log(log_entry, DB_FILE)
    response = {"food": food_name, "confidence": float(confidence), "nutrition": nutrition, "log_entry": log_entry}
    return jsonify(response), 200

@app.route("/logs", methods=["GET"])
def get_logs():
    logs = read_logs(DB_FILE)
    return jsonify({"logs": logs})

@app.route("/weekly_report", methods=["GET"])
def weekly_report():
    logs = read_logs(DB_FILE)
    one_week_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    week_logs = [log for log in logs if datetime.datetime.fromisoformat(log["timestamp"]) >= one_week_ago]
    total_calories = sum((log.get("nutrition") or {}).get("calories", 0) for log in week_logs)
    # simple aggregates
    by_food = {}
    for l in week_logs:
        f = l["food"]
        by_food[f] = by_food.get(f, 0) + 1
    return jsonify({
        "total_calories": total_calories,
        "meals_logged": len(week_logs),
        "top_foods": sorted(by_food.items(), key=lambda x: -x[1])[:10],
        "logs": week_logs
    })

@app.route("/predict_sickness", methods=["GET"])
def predict_sickness_route():
    logs = read_logs(DB_FILE)
    risk, risky_foods, recommendations = predict_sickness(logs, model=SICKNESS_MODEL)
    return jsonify({"risk": float(risk), "risky_foods": risky_foods, "recommendations": recommendations})

@app.route("/recommendations", methods=["GET"])
def recommendations_route():
    logs = read_logs(DB_FILE)
    # reuse sickness predictor for recommendations
    _, risky_foods, recs = predict_sickness(logs, model=SICKNESS_MODEL)
    # add generalized caloric advice
    avg_cal = None
    week_logs = [log for log in logs if datetime.datetime.fromisoformat(log["timestamp"]) >= datetime.datetime.utcnow()-datetime.timedelta(days=7)]
    if week_logs:
        avg_cal = sum((log.get("nutrition") or {}).get("calories",0) for log in week_logs)/len(week_logs)
    payload = {"risky_foods": risky_foods, "recommendations": recs, "average_weekly_cal": avg_cal}
    return jsonify(payload)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''

#after integrating with food_classifier.py
'''
import os
import uuid
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

#from backend.utils import get_nutrition_by_food, append_log, read_logs
#from backend.food_classifier import load_model_and_labels, predict_food_from_path
from backend.models.food_classifier import load_model_and_labels, predict_food_from_path
from backend.utils import get_nutrition_by_food, append_log, read_logs

# Flask setup
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

# Load model & labels once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "food_classifier.h5")
LABELS, MODEL = load_model_and_labels(MODEL_PATH)

# Database & calorie files
DB_FILE = os.path.join(os.path.dirname(__file__), "db.json")
CAL_DB_FILE = os.path.join(os.path.dirname(__file__), "food_calories.json")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save file
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Predict
    food_name, confidence = predict_food_from_path(save_path, MODEL, LABELS)
    nutrition = get_nutrition_by_food(food_name, CAL_DB_FILE) or {}

    # Log
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "source_file": unique_name
    }
    append_log(log_entry, DB_FILE)

    return jsonify({
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    }), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    logs = read_logs(DB_FILE)
    return jsonify({"logs": logs})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''
#after integrating with frontend and model

import os
import uuid
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import backend modules
from backend.utils import get_nutrition_by_food, append_log, read_logs, get_weekly_risks
from backend.models.food_classifier import load_model_and_labels, predict_food_from_path

# --------------------- Flask setup --------------------- #
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

# ---------------- Load model & labels ------------------ #
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "food_classifier.h5")
LABELS, MODEL = load_model_and_labels(MODEL_PATH)

# ---------------- DB / Calorie files ------------------- #
DB_FILE = os.path.join(os.path.dirname(__file__), "db.json")
CAL_FILE = os.path.join(os.path.dirname(__file__), "food_calories.json")


# ---------------- Helper ------------------------------ #
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------- Routes ------------------------------ #
@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save uploaded image
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Predict food
    food_name, confidence = predict_food_from_path(save_path, MODEL, LABELS)

    # Get nutrition info
    nutrition = get_nutrition_by_food(food_name, CAL_FILE) or {}

    # Append log entry
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "source_file": unique_name
    }
    append_log(log_entry, DB_FILE)

    return jsonify({
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    })


@app.route("/logs", methods=["GET"])
def get_logs():
    logs = read_logs(DB_FILE)
    return jsonify({"logs": logs})


@app.route("/predict_sick", methods=["GET"])
def predict_sick():
    """
    Simple risk assessment: example logic - foods with calories > 700 are risky.
    Returns list of risky foods and advice.
    """
    logs = read_logs(DB_FILE)
    risky_foods = [entry["food"] for entry in logs if entry.get("nutrition", {}).get("calories", 0) > 700]
    advice = get_weekly_risks(risky_foods)
    return jsonify({"risky_foods": risky_foods, "advice": advice})


# ---------------- Run App ----------------------------- #
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
