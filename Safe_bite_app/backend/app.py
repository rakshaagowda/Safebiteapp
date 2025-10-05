# backend/app.py
'''import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import datetime
import json
from calorie_scanner import predict_food
#from utils import get_nutrition_by_food, append_log
from folder_name.utils import get_nutrition_by_food, append_log
UPLOAD_FOLDER = os.path.join("backend", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)  # allow cross-origin requests for local dev

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        food_name, confidence = predict_food(save_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    nutrition = get_nutrition_by_food(food_name)
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition or {},
        "source_file": filename
    }
    append_log(log_entry)

    response = {
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    }
    return jsonify(response), 200

@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        with open(os.path.join("backend", "db.json"), "r", encoding="utf-8") as f:
            return jsonify({"logs": json.load(f)})
    except Exception:
        return jsonify({"logs": []})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''
# backend/app.py
'''
import sys
import os
import datetime
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure current directory is in Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports (Pylance-safe)
from calorie_scanner import predict_food  # type: ignore
from utils import get_nutrition_by_food, append_log  # type: ignore

# Upload folder setup
UPLOAD_FOLDER = os.path.join("backend", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)  # allow cross-origin requests for local dev

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        food_name, confidence = predict_food(save_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    nutrition = get_nutrition_by_food(food_name) or {}
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "source_file": filename
    }
    append_log(log_entry)

    response = {
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    }
    return jsonify(response), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    db_file = os.path.join("backend", "db.json")
    try:
        if not os.path.exists(db_file):
            return jsonify({"logs": []})
        with open(db_file, "r", encoding="utf-8") as f:
            return jsonify({"logs": json.load(f)})
    except Exception as e:
        return jsonify({"logs": [], "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''

# backend/app.py
'''
import os
import sys
import datetime
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure current directory is in Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports (Pylance-safe)
from calorie_scanner import predict_food  # type: ignore
#from utils import get_nutrition_by_food, append_log  
from utils import get_nutrition_by_food, append_log


# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_FILE = os.path.join(BASE_DIR, "db.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Generate unique filename
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        food_name, confidence = predict_food(save_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    nutrition = get_nutrition_by_food(food_name) or {}
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "source_file": unique_name
    }

    append_log(log_entry)  # Make sure this appends safely to DB_FILE

    response = {
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    }
    return jsonify(response), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        if not os.path.exists(DB_FILE):
            return jsonify({"logs": []})
        with open(DB_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return jsonify({"logs": logs})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"logs": [], "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    '''
# backend/app.py
'''
import os
import sys
import datetime
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure current directory is in Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports (Pylance-safe)
#from calorie_scanner import predict_food  
#from utils import get_nutrition_by_food, append_log  
from backend.utils import get_nutrition_by_food, append_log
from backend.calorie_scanner import predict_food



# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_FILE = os.path.join(BASE_DIR, "db.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}


def allowed_file(filename: str | None) -> bool:
    """Check if filename has an allowed extension."""
    if filename is None:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    # Type narrowing: check if filename exists and is not empty
    if not file.filename or file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # At this point, file.filename is guaranteed to be a non-empty string
    # Generate unique filename
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        food_name, confidence = predict_food(save_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    nutrition = get_nutrition_by_food(food_name) or {}
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "source_file": unique_name
    }

    append_log(log_entry)  # Make sure this appends safely to DB_FILE

    response = {
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    }
    return jsonify(response), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        if not os.path.exists(DB_FILE):
            return jsonify({"logs": []})
        with open(DB_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return jsonify({"logs": logs})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"logs": [], "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    '''
# backend/app.py

import os
import sys
import datetime
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure current directory is in Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from backend.utils import get_nutrition_by_food, append_log
from backend.calorie_scanner import predict_food  # your ML prediction function

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_FILE = os.path.join(BASE_DIR, "db.json")
CAL_DB_FILE = os.path.join(BASE_DIR, "food_calories.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}


def allowed_file(filename: str | None) -> bool:
    """Check if filename has an allowed extension."""
    if filename is None:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/scan_food", methods=["POST"])
def scan_food():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if not file.filename or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        food_name, confidence = predict_food(save_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    nutrition = get_nutrition_by_food(food_name, CAL_DB_FILE) or {}
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "source_file": unique_name
    }

    append_log(log_entry, DB_FILE)

    response = {
        "food": food_name,
        "confidence": confidence,
        "nutrition": nutrition,
        "log_entry": log_entry
    }
    return jsonify(response), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        if not os.path.exists(DB_FILE):
            return jsonify({"logs": []})
        with open(DB_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return jsonify({"logs": logs})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"logs": [], "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
