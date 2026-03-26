import os
import time
import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pickle
import wfdb
import uuid
from pymongo import MongoClient

# Auth controller (Ensure auth_controller.py is also updated for MongoDB!)
from auth_controller import signup, login

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) 

# --- Directory Configuration ---
UPLOAD_FOLDER = 'uploads/'
TEMP_FOLDER = 'temp/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MongoDB Connection ---
try:
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[os.environ.get("DB_DATABASE", "hospital_automanager")]
    
    # Test connection
    client.server_info() 
    print("✓ MongoDB connected successfully.")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}. DB routes will return 503 errors.")
    db = None

# --- LOAD MACHINE LEARNING MODEL ---
try:
    model = tf.keras.models.load_model("ecg_model.h5")
    print("✓ Deep Learning Model loaded successfully!")
    
    with open("ecg_classes.pkl", "rb") as f:
        CLASSES = pickle.load(f)
    print(f"✓ Classes loaded from pickle: {CLASSES}")
    
except Exception as e:
    print(f"⚠️ Warning: Could not load model or classes. Error: {e}")
    model = None
    CLASSES = ['CD','HYP','MI','NORM','STTC'] # Fallback classes


# ==========================================
# -------------- AUTH ROUTES ---------------
# ==========================================
@app.route('/api/auth/signup', methods=['POST'])
def signup_route():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    return signup(request, db)

@app.route('/api/auth/login', methods=['POST'])
def login_route():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    return login(request, db)


# ==========================================
# --------- MEDICAL HISTORY ROUTES ---------
# ==========================================
@app.route('/api/medical-history/<int:user_id>', methods=['GET'])
def get_medical_history(user_id):
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    try:
        # find_one finds the document, {"_id": 0} hides the MongoDB specific Object ID from React
        result = db.medical_history.find_one({"user_id": str(user_id)}, {"_id": 0})
        return jsonify(result) if result else jsonify(None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/medical-history', methods=['POST'])
def save_medical_history():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    data = request.json

    try:
        update_data = {
            "user_id": str(data.get('userId')),
            "age": data.get('age'),
            "sex": data.get('sex'),
            "family_heart_history": 1 if data.get('familyHistory') == 'yes' else 0,
            "past_heart_problem": data.get('pastHeartProblem')
        }
        
        # Upsert=True means "Update if exists, Create if it doesn't"
        db.medical_history.update_one(
            {"user_id": str(data.get('userId'))},
            {"$set": update_data},
            upsert=True
        )
        return jsonify({"message": "Medical history saved successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# --------- LIFESTYLE DATA ROUTES ----------
# ==========================================
@app.route('/api/lifestyle-data/<int:user_id>', methods=['GET'])
def get_lifestyle_data(user_id):
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    try:
        db_data = db.lifestyle_data.find_one({"user_id": str(user_id)}, {"_id": 0})
        
        if db_data:
            def safe_str(val):
                return str(val) if val is not None else ""

            return jsonify({
                "physicalActivity": db_data.get("physicalActivity", ""),
                "smoking": db_data.get("smoking", ""),
                "alcoholUse": db_data.get("alcoholUse", ""),
                "otherSubstances": db_data.get("otherSubstances", ""),
                "chestDiscomfort": safe_str(db_data.get("chestDiscomfort")),
                "exerciseAngina": safe_str(db_data.get("exerciseAngina"))
            })
        return jsonify(None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lifestyle-data', methods=['POST'])
def save_lifestyle_data():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    data = request.json
    user_id = str(data.get('userId'))

    try:
        update_fields = {}
        
        # Only add text fields to the update list if they aren't empty
        for key in ['physicalActivity', 'smoking', 'alcoholUse', 'otherSubstances']:
            if data.get(key):
                update_fields[key] = data.get(key)
                
        # Only add integer fields to the update list if they aren't empty
        for key in ['chestDiscomfort', 'exerciseAngina']:
            val = data.get(key)
            if val != "" and val is not None:
                update_fields[key] = int(val)

        if update_fields:
            # $set acts like COALESCE automatically. It won't overwrite existing fields not sent in the update.
            db.lifestyle_data.update_one(
                {"user_id": user_id},
                {"$set": update_fields},
                upsert=True
            )
        
        return jsonify({"message": "Data saved successfully!"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# --------- CLINICAL & FILE UPLOAD ---------
# ==========================================
@app.route('/api/upload-report', methods=['POST'])
def upload_report():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    
    user_id = str(request.form.get('userId'))

    try:
        # Build the document
        report_doc = {
            "user_id": user_id,
            "report_type": request.form.get('reportType'),
            "resting_bp": request.form.get('restingBP') or None,
            "cholesterol": request.form.get('cholesterol') or None,
            "fasting_bs": request.form.get('fastingBS') or None,
            "max_hr": request.form.get('maxHR') or None,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc)
        }

        # Handle the file upload (if they selected one)
        if 'report' in request.files and request.files['report'].filename != '':
            file = request.files['report']
            filename = f"{int(time.time() * 1000)}-{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            report_doc["file_path"] = file_path # Add file path to document

        # Insert new record into MongoDB
        db.health_reports.insert_one(report_doc)
        
        return jsonify({
            "message": "Clinical data and/or file saved successfully!", 
            "filePath": report_doc.get("file_path")
        }), 200

    except Exception as e:
        print("Database Error:", str(e))
        return jsonify({"error": "Failed to save data", "details": str(e)}), 500


# ==========================================
# ------- ML LIVE PREDICTION ROUTE ---------
# ==========================================
@app.route('/api/analyze-live-ecg', methods=['POST'])
def analyze_live_ecg():
    data = request.json
    raw_data = data.get('ecgData')

    if not raw_data or len(raw_data) == 0:
        return jsonify({"error": "No ECG data received"}), 400
    if model is None:
        return jsonify({"error": "Machine Learning model is not loaded on the server"}), 500

    try:
        signal = np.array(raw_data, dtype=float)
        
        # Crop or Pad to 1000 points
        if len(signal) > 1000:
            signal = signal[:1000] 
        else:
            signal = np.pad(signal, (0, 1000 - len(signal)), 'constant')
            
        # Model expects 3 leads (1000, 3)
        final_signal = np.zeros((1000, 3))
        final_signal[:, 0] = signal
        
        # Normalize and Predict
        X = np.expand_dims(final_signal, 0)
        X = (X - X.mean()) / (X.std() + 1e-8) 
        pred = model.predict(X)[0]
        
        result = dict(zip(CLASSES, [float(p) for p in pred]))
        best_diagnosis_code = max(result, key=result.get)
        
        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }
        
        return jsonify({
            "message": "Success",
            "diagnosis": diagnosis_names.get(best_diagnosis_code, best_diagnosis_code),
            "confidence": result[best_diagnosis_code],
            "all_probabilities": result
        }), 200
    except Exception as e:
        return jsonify({"error": "Failed to process ECG data", "details": str(e)}), 500

# ==========================================
# ------- WFDB (.dat & .hea) PREDICTION ----
# ==========================================
@app.route('/api/analyze-wfdb', methods=['POST'])
def analyze_wfdb():
    if 'dat' not in request.files or 'hea' not in request.files:
        return jsonify({"error": "Please upload BOTH the .dat and .hea files"}), 400

    if model is None:
        return jsonify({"error": "Machine Learning model is not loaded"}), 500

    uid = str(uuid.uuid4())
    dat_path = os.path.join(TEMP_FOLDER, f"{uid}.dat")
    hea_path = os.path.join(TEMP_FOLDER, f"{uid}.hea")
    base_path = os.path.join(TEMP_FOLDER, uid)

    try:
        request.files['dat'].save(dat_path)
        request.files['hea'].save(hea_path)
        
        signal, fields = wfdb.rdsamp(base_path)

        if len(signal) > 1000:
            signal = signal[:1000]
        else:
            signal = np.pad(signal, ((0, 1000 - len(signal)), (0, 0)), 'constant')

        final_signal = np.zeros((1000, 3))
        leads_to_copy = min(signal.shape[1], 3)
        final_signal[:, :leads_to_copy] = signal[:, :leads_to_copy]

        X = np.expand_dims(final_signal, 0)
        X = (X - X.mean()) / (X.std() + 1e-8) 
        pred = model.predict(X)[0]

        result = dict(zip(CLASSES, [float(p) for p in pred]))
        best_diagnosis_code = max(result, key=result.get)

        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }

        # Cleanup
        os.remove(dat_path)
        os.remove(hea_path)

        return jsonify({
            "message": "Success",
            "diagnosis": diagnosis_names.get(best_diagnosis_code, best_diagnosis_code),
            "confidence": result[best_diagnosis_code],
            "sampling_rate": fields.get('fs', 'Unknown')
        }), 200

    except Exception as e:
        if os.path.exists(dat_path): os.remove(dat_path)
        if os.path.exists(hea_path): os.remove(hea_path)
        return jsonify({"error": "Failed to process ECG files. Ensure valid WFDB format.", "details": str(e)}), 500

# Start Server
if __name__ == '__main__':
    # Render provides a $PORT environment variable. If it's missing, default to 5000.
    port = int(os.environ.get("PORT", 5000))
    print(f"Flask Server running on port {port}")
    app.run(host='0.0.0.0', port=port)