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
import jwt
from pymongo import MongoClient, ReturnDocument
import cloudinary
import cloudinary.uploader
from flask_socketio import SocketIO, emit
# Auth controller (Ensure auth_controller.py is also updated for MongoDB!)
from auth_controller import signup, login, card_login # <-- Add it to the import


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) 

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# A dictionary to hold the 1000 data points while they stream in
active_ecg_sessions = {}

# --- Directory Configuration ---
UPLOAD_FOLDER = 'uploads/'
TEMP_FOLDER = 'temp/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Grab the exact keys you just pasted
my_cloud_name = os.environ.get('CLOUD_NAME')
my_api_key = os.environ.get('CLOUD_API_KEY')
my_api_secret = os.environ.get('CLOUD_API_SECRET')

# Diagnostic Print - This will show up in your terminal!
print("=========================================")
print(f"☁️  CLOUD NAME: {my_cloud_name}")
print(f"🔑 API KEY: {my_api_key}")
print("=========================================")

# Configure Cloudinary
cloudinary.config( 
  cloud_name = my_cloud_name, 
  api_key = my_api_key, 
  api_secret = my_api_secret,
  secure = True
)
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
# ==========================================
# LOAD ML MODEL (UPDATED FIX)
# ==========================================
import os
import joblib
import numpy as np
import tensorflow as tf

import os
import joblib
import tensorflow as tf

base_dir = os.path.dirname(__file__)
# Use an 'r' before the string so Windows backslashes don't break the path
MODEL_PATH = os.path.join(base_dir, 'model_1lead_500hz.h5')
X_SCALER_PATH = os.path.join(base_dir, 'X_scaler_500hz.pkl')
Y_SCALER_PATH = os.path.join(base_dir, 'Y_scaler_stats_500hz.pkl')

import os
import joblib
import numpy as np

# ==========================================
# LOAD BLOOD AI MODEL & SCALER
# ==========================================
try:
    print("🧠 Loading Blood Biomarker Model...", flush=True)
    
    # 1. Find the exact folder where this app.py file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the exact file paths
    blood_scaler_path = os.path.join(base_dir, 'scaler.pkl')
    blood_model_path = os.path.join(base_dir, 'svm_model.pkl')
    
    # 3. Load the files using the secure paths
    blood_scaler = joblib.load(blood_scaler_path)
    blood_model = joblib.load(blood_model_path)
    
    print("✓ Blood AI Model & Scaler loaded successfully!", flush=True)
    
except Exception as e:
    print(f"⚠️ Blood AI Initialization Error: {e}", flush=True)
    blood_scaler = None
    blood_model = None

print("--- DIAGNOSTICS ---")
print(f"Can Python see the Model? : {os.path.exists(MODEL_PATH)}")
print(f"Can Python see X_Scaler?: {os.path.exists(X_SCALER_PATH)}")
print(f"Can Python see Y_Scaler?: {os.path.exists(Y_SCALER_PATH)}")
print("-------------------")

try:
    # Only try to load if the file actually exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Python cannot find the file at: {MODEL_PATH}")

    print("🧠 Loading Deep Learning Model and Scalers...", flush=True)
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    X_scaler = joblib.load(X_SCALER_PATH)              
    Y_scaler_stats = joblib.load(Y_SCALER_PATH) 
    
    Y_mean = Y_scaler_stats['mean']
    Y_std = Y_scaler_stats['std']
    
    CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    print("✓ Model and Scalers loaded and online!", flush=True)
    
except Exception as e:
    print(f"⚠️ Initialization Error: {e}", flush=True)
    model = None
    X_scaler = None
# ==========================================
# -------------- AUTH ROUTES ---------------
# ==========================================
import jwt
from functools import wraps
from flask import request, jsonify
from bson.objectid import ObjectId

# --- JWT Authentication Middleware ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Check if token is in the headers
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] # Bearer <token>
            
        if not token:
            return jsonify({"error": "Token is missing!"}), 401
            
        try:
            # Decode the token (Make sure 'YOUR_SECRET_KEY' matches your login route!)
            data = jwt.decode(token, os.getenv('JWT_SECRET', 'YOUR_SECRET_KEY'), algorithms=["HS256"])
            # Fetch the user from MongoDB
            current_user = db.users.find_one({"_id": ObjectId(data["user_id"])})
            
            if not current_user:
                return jsonify({"error": "User not found!"}), 401
                
        except Exception as e:
            return jsonify({"error": "Token is invalid!", "details": str(e)}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated


# --- Profile GET Route ---
@app.route('/api/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    try:
        # Return the user data (Exclude the password!)
        user_profile = {
            "id": str(current_user.get("_id")),
            "firstName": current_user.get("first_name", "Unknown"),
            "lastName": current_user.get("last_name", ""),
            "email": current_user.get("email", "No Email Provided"),
            "mobile": current_user.get("mobile", "No Phone Provided"), # Added Mobile!
            "role": current_user.get("role", "Patient"), # Changed to Patient
            "cardId": current_user.get("card_id", "Not Linked")
        }
        
        return jsonify({
            "status": "success",
            "user": user_profile
        }), 200
        
    except Exception as e:
        print(f"❌ Profile Fetch Error: {e}", flush=True)
        return jsonify({"error": "Failed to fetch profile"}), 500
    
# ==========================================
# --------- BLOOD BIOMARKER ROUTE ----------
# ==========================================
import datetime

# ==========================================
# --------- BLOOD BIOMARKER ROUTE ----------
# ==========================================
# ==========================================
# BLOOD AI PREDICTION ROUTE (UPDATED)
# ==========================================
@app.route('/api/analyze-blood', methods=['POST'])
def analyze_blood():
    data = request.json
    user_id = str(data.get('userId', 'guest'))
    
    if user_id == 'guest':
        return jsonify({"error": "Patient must be logged in to analyze blood data."}), 400

    try:
        # 1. FETCH DATA FROM MONGODB (We don't rely on React for the numbers anymore!)
        if 'db' not in globals() or db is None:
            return jsonify({"error": "Database unavailable"}), 503
            
        user_data = db.medical_history.find_one({"user_id": user_id})
        
        if not user_data:
            return jsonify({"error": "No medical history found. Please fill out patient profile first."}), 404

        # 2. Extract Biomarkers from the Database (Fallback to 0.0 if empty)
        troponin = float(user_data.get('troponin') or 0.0)
        ck_mb = float(user_data.get('ck_mb') or 0.0)
        bnp = float(user_data.get('bnp') or 0.0)
        potassium = float(user_data.get('potassium') or 0.0)
        creatinine = float(user_data.get('creatinine') or 0.0)
        
        # 3. Safety Check: Did they actually enter blood data?
        if all(v == 0.0 for v in [troponin, ck_mb, bnp, potassium, creatinine]):
            return jsonify({"error": "No blood biomarker data found in patient's Medical History."}), 400

        print(f"🩸 DB Data Pulled: Trop={troponin}, CK-MB={ck_mb}, BNP={bnp}, K={potassium}, Cr={creatinine}", flush=True)

        # 4. Check Models
        if blood_scaler is None or blood_model is None:
            return jsonify({"error": "Blood AI models are offline on the server."}), 500

        # 5. Format, Scale, and Predict
        patient_features = np.array([[troponin, ck_mb, bnp, potassium, creatinine]])
        scaled_features = blood_scaler.transform(patient_features)
        
        prediction_code = blood_model.predict(scaled_features)[0]
        probabilities = blood_model.predict_proba(scaled_features)[0]
        confidence = round(max(probabilities) * 100, 2)

        # 6. Map Diagnosis
        if str(prediction_code) == "1" or str(prediction_code).lower() == "myocardial infarction":
            final_diagnosis = "High Risk - Myocardial Infarction Detected"
        else:
            final_diagnosis = "Normal Blood Biomarkers"

        # 7. Update Record (We save the prediction back into their medical history)
        # ... (Previous code)
        # 7. Update Record (We save the prediction back into their medical history)
        db.medical_history.update_one(
            {"user_id": user_id},
            {"$set": {
                "latest_blood_diagnosis": final_diagnosis,
                "latest_blood_confidence": confidence,
                "last_blood_analysis_time": datetime.datetime.now(datetime.timezone.utc)
            }}
        )
        print(f"💾 Blood prediction successfully saved to MongoDB for user {user_id}", flush=True)

        # ---> ADD THE BIOMARKERS TO THE RESPONSE HERE <---
        return jsonify({
            "status": "success",
            "diagnosis": final_diagnosis,
            "confidence": confidence,
            "biomarkers": {
                "troponin": troponin,
                "ck_mb": ck_mb,
                "bnp": bnp,
                "potassium": potassium,
                "creatinine": creatinine
            }
        }), 200

    except Exception as e:
        print(f"❌ Blood Analysis Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
# ==========================================
# --------- ECG HISTORY ROUTE --------------
# ==========================================


# ==========================================
# --------- BLOOD HISTORY ROUTE ------------
# ==========================================
@app.route('/api/blood-records/<string:user_id>', methods=['GET'])
def get_blood_records(user_id):
    if 'db' not in globals() or db is None: 
        return jsonify({"error": "Database unavailable"}), 503
        
    try:
        # Fetch records, sort by newest first (-1), and hide the MongoDB Object ID
        records = list(db.blood_records.find(
            {"user_id": str(user_id)}, 
            {"_id": 0}
        ).sort("timestamp", -1))
        
        # Convert Datetime objects into strings for React
        for rec in records:
            if 'timestamp' in rec and rec['timestamp']:
                rec['timestamp'] = rec['timestamp'].isoformat()
                
        return jsonify({
            "status": "success", 
            "records": records
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching blood records: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/ecg-records/<string:user_id>', methods=['GET'])
def get_ecg_records(user_id):
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    try:
        # Fetch all records for this user, sorted by timestamp (-1 means newest first)
        records = list(db.ecg_records.find(
            {"user_id": str(user_id)}, 
            {"_id": 0} # Hide the MongoDB ObjectID so it doesn't break React
        ).sort("timestamp", -1))
        
        # Convert complex Datetime objects into normal strings for the frontend
        for rec in records:
            if 'timestamp' in rec:
                rec['timestamp'] = rec['timestamp'].isoformat()
                
        return jsonify({
            "status": "success", 
            "records": records
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching ECG records: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
@app.route('/api/auth/signup', methods=['POST'])
def signup_route():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    return signup(request, db)

@app.route('/api/auth/login', methods=['POST'])
def login_route():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    return login(request, db)

@app.route('/api/auth/card-login', methods=['POST'])
def card_login_route():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    return card_login(request, db)


# ==========================================
# --------- MEDICAL HISTORY ROUTES ---------
# ==========================================
@app.route('/api/medical-history/<string:user_id>', methods=['GET'])
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
    troponin = float(data.get('troponin')) if data.get('troponin') else 0
    ck_mb = float(data.get('ck_mb')) if data.get('ck_mb') else 5
    bnp = float(data.get('bnp')) if data.get('bnp') else 2
    potassium = float(data.get('potassium')) if data.get('potassium') else 0.2
    creatinine = float(data.get('creatinine')) if data.get('creatinine') else 0.2
    try:
        update_data = {
            "user_id": str(data.get('userId')),
            "age": data.get('age'),
            "sex": data.get('sex'), 
            
            # --- New AI Metadata Fields ---
            "height": data.get('height'),
            "weight": data.get('weight'),
            "inf1": 1 if data.get('inf1') == 'yes' else 0,
            "inf2": 1 if data.get('inf2') == 'yes' else 0,
            "pace": 1 if data.get('pace') == 'yes' else 0,
            "extra": 1 if data.get('extra') == 'yes' else 0,
            
            # --- Original Fields ---
            "family_heart_history": 1 if data.get('familyHistory') == 'yes' else 0,
            "past_heart_problem": data.get('pastHeartProblem'),
            "troponin": troponin,
            "ck_mb": ck_mb,
            "bnp": bnp,
            "potassium": potassium,
            "creatinine": creatinine
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
@app.route('/api/lifestyle-data/<string:user_id>', methods=['GET'])
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
        
        # CHANGED: Treat ALL of these as text/string fields now! No more integers.
        all_fields = [
            'physicalActivity', 'smoking', 'alcoholUse', 
            'otherSubstances', 'chestDiscomfort', 'exerciseAngina'
        ]
        
        for key in all_fields:
            val = data.get(key)
            if val != "" and val is not None:
                update_fields[key] = val  # We removed the int(val) conversion here

        if update_fields:
            # $set won't overwrite existing fields not sent in the update
            db.lifestyle_data.update_one(
                {"user_id": user_id},
                {"$set": update_fields},
                upsert=True
            )
        
        return jsonify({"message": "Data saved successfully!"}), 200
        
    except Exception as e:
        # I added this print statement so you can see exact errors in your Flask terminal!
        print(f"Backend Crash in lifestyle-data: {str(e)}") 
        return jsonify({"error": str(e)}), 500

@app.route('/api/health-report/<string:user_id>', methods=['GET'])
def get_health_report(user_id):
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    try:
        # Find the most recent blood report for this user
        result = db.health_reports.find_one(
            {"user_id": str(user_id), "report_type": "blood_report"},
            sort=[("uploaded_at", -1)], # -1 gets the newest one if they uploaded multiple
            projection={"_id": 0} # Hides the MongoDB ObjectID
        )
        return jsonify(result) if result else jsonify(None)
    except Exception as e:
        print(f"Error fetching health report: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
# ==========================================
# --------- CLINICAL & FILE UPLOAD ---------
# ==========================================
# ==========================================
# --------- CLINICAL & CLOUD UPLOAD --------
# ==========================================
@app.route('/api/upload-report', methods=['POST'])
def upload_report():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    
    user_id = str(request.form.get('userId'))

    try:
        # Build the MongoDB document
        report_doc = {
            "user_id": user_id,
            "report_type": request.form.get('reportType'),
            "resting_bp": request.form.get('restingBP') or None,
            "cholesterol": request.form.get('cholesterol') or None,
            "fasting_bs": request.form.get('fastingBS') or None,
            "max_hr": request.form.get('maxHR') or None,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc)
        }

        # Catch the file and send it straight to Cloudinary!
        if 'report' in request.files and request.files['report'].filename != '':
            file = request.files['report']
            
            # This single line uploads it to the cloud
            upload_result = cloudinary.uploader.upload(
                file, 
                folder="hospital_reports",
                resource_type="auto" # Automatically handles images, PDFs, etc.
            )
            
            # Grab the permanent HTTPS link and save it to the database document
            report_doc["file_url"] = upload_result.get("secure_url")

        # Insert the record (with the cloud link) into MongoDB
        db.health_reports.insert_one(report_doc)
        
        # We use .get() here in case they didn't upload a file and only saved vitals
        return jsonify({
            "message": "Clinical data saved successfully!", 
            "fileUrl": report_doc.get("file_url")
        }), 200

    except Exception as e:
        print("Database/Cloud Error:", str(e))
        return jsonify({"error": "Failed to save data", "details": str(e)}), 500


import wfdb
import numpy as np
import os
from scipy.signal import butter, filtfilt

# ==========================================
# 1. DIGITAL FILTER FUNCTION
# ==========================================
def apply_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500.0, order=3):
    """Flattens breathing wander and removes 50/60Hz electrical fuzz."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data) 
    return filtered_data
# ==========================================
# ------- ML LIVE PREDICTION ROUTE ---------
# ==========================================
@app.route('/api/analyze-live-ecg', methods=['POST'])
def analyze_live_ecg():
    data = request.json
    raw_data = data.get('ecgData')

    if not raw_data or len(raw_data) == 0:
        return jsonify({"error": "No ECG data received"}), 400
        
    # Ensure both the model and scalers are online
    if model is None or X_scaler is None:
        return jsonify({"error": "Machine Learning model or scalers are not loaded on the server"}), 500

    try:
        # 1. Crop or Pad to exactly 5000 points (10 seconds @ 500Hz)
        TARGET_LENGTH = 5000
        signal = np.array(raw_data, dtype=float)
        
        if len(signal) > TARGET_LENGTH:
            signal = signal[:TARGET_LENGTH] 
        elif len(signal) < TARGET_LENGTH:
            signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)), 'constant')
            
        # ==========================================
        # 2. THE MISSING PHYSICS & FILTER BLOCK
        # ==========================================
        # Convert ADC (0-4095) to Volts (3.3V)
        voltage = (signal / 4095.0) * 3.3
        
        # Center at 0.0
        centered_voltage = voltage - np.mean(voltage)
        
        # Convert to true Millivolts (1100x Gain)
        ecg_mv = centered_voltage / 1.1
        
        # Apply the SciPy Butterworth Filter (Removes baseline wander & noise)
        clean_ecg_mv = apply_bandpass_filter(ecg_mv, lowcut=0.5, highcut=40.0, fs=500.0)

        # 3. Prepare the Signal (Y) -> Shape: (1, 5000, 1)
        Y_input = np.reshape(clean_ecg_mv, (1, TARGET_LENGTH, 1))
        
        # Scale using your pre-trained global stats from the .pkl file
        Y_input_scaled = (Y_input - Y_mean) / Y_std 
        
        # 4. Prepare the Metadata (X) -> Shape: (1, 7)
        dummy_metadata = np.array([[50, 1, 170, 70, 0, 0, 0]])
        X_input_scaled = X_scaler.transform(dummy_metadata)
        
        # 5. Normalize and Predict (Pass BOTH inputs to the multi-modal model)
        pred = model.predict([X_input_scaled, Y_input_scaled], verbose=0)[0]
        
        # 6. Format the Results
        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }
        
        raw_result = dict(zip(CLASSES, [float(p) for p in pred]))
        best_diagnosis_code = max(raw_result, key=raw_result.get)
        
        # Convert ALL 5 probabilities into clean percentages with READABLE names
        percentages_result = {}
        for code, prob in raw_result.items():
            readable_name = diagnosis_names.get(code, code)
            percentages_result[readable_name] = round(prob * 100, 2)
        
        return jsonify({
            "message": "Success",
            "diagnosis": diagnosis_names.get(best_diagnosis_code, best_diagnosis_code),
            "confidence": percentages_result[diagnosis_names.get(best_diagnosis_code)],
            "all_probabilities": percentages_result
        }), 200
        
    except Exception as e:
        print(f"Live ECG Processing Error: {e}", flush=True)
        return jsonify({"error": "Failed to process ECG data", "details": str(e)}), 500
# ==========================================
# ------- WFDB (.dat & .hea) PREDICTION ----
# ==========================================
import tempfile
import numpy as np

# ==========================================
# ------- WFDB (.dat & .hea) PREDICTION ----
# ==========================================
import numpy as np
import tempfile
import os
import wfdb
from flask import jsonify, request
from werkzeug.utils import secure_filename

@app.route('/api/analyze-wfdb', methods=['POST'])
def analyze_wfdb():
    if 'hea' not in request.files or 'dat' not in request.files:
        return jsonify({"error": "Missing .hea or .dat file"}), 400

    hea_file = request.files['hea']
    dat_file = request.files['dat']
    
    # Grab the userId sent from the React frontend (we will need to append this in React!)
    user_id = request.form.get('userId') 
    print(f"✅ Successfully read {user_id}!", flush=True)

    # 1. Create a secure temporary directory that auto-deletes
    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = os.path.splitext(secure_filename(hea_file.filename))[0]
        
        hea_path = os.path.join(temp_dir, f"{base_name}.hea")
        dat_path = os.path.join(temp_dir, f"{base_name}.dat")
        
        hea_file.save(hea_path)
        dat_file.save(dat_path)
        
        try:
            # 2. Read the WFDB file
            record_path = os.path.join(temp_dir, base_name)
            record = wfdb.rdsamp(record_path)
            
            # Extract the raw signal array
            signal_data = record[0][:, 0] 
            
            print(f"✅ Successfully read {len(signal_data)} samples from {base_name}.hea!", flush=True)

            # 3. Format exactly 5000 samples for the 500Hz AI Model
            TARGET_LENGTH = 5000
            
            if len(signal_data) > TARGET_LENGTH:
                signal_data = signal_data[:TARGET_LENGTH] # Truncate if too long
            elif len(signal_data) < TARGET_LENGTH:
                # Pad with zeros if it's too short
                signal_data = np.pad(signal_data, (0, TARGET_LENGTH - len(signal_data)), 'constant')

            # 4. AI Prediction Block
            if model is not None and X_scaler is not None:
                
                # --- A. PREPARE THE SIGNAL (Y) ---
                Y_input = np.reshape(signal_data, (1, TARGET_LENGTH, 1))
                Y_input_scaled = (Y_input - Y_mean) / Y_std 
                
                # --- B. PREPARE THE METADATA (X) FROM DATABASE ---
                # Step 1: Set default baseline values (Fallback)
                age, sex, height, weight, inf1, inf2, pace = 50.0, 1.0, 170.0, 70.0, 0.0, 0.0, 0.0
                
                # Step 2: Try to fetch the real data from MongoDB
                if db is not None and user_id:
                    try:
                        user_data = db.medical_history.find_one({"user_id": str(user_id)})
                        if user_data:
                            # Safely extract and cast data from DB, using defaults if fields are empty
                            age = float(user_data.get('age') or age)
                            sex_val = user_data.get('sex', 'male')
                            sex = 1.0 if str(sex_val).lower() == 'male' else 0.0
                            height = float(user_data.get('height') or height)
                            weight = float(user_data.get('weight') or weight)
                            inf1 = float(user_data.get('inf1') or inf1)
                            inf2 = float(user_data.get('inf2') or inf2)
                            pace = float(user_data.get('pace') or pace)
                            
                            print(f"📊 Loaded DB Profile for user {user_id}: Age={age}, Sex={sex_val}", flush=True)
                        else:
                            print(f"⚠️ No DB profile found for user {user_id}, using defaults.", flush=True)
                    except Exception as db_err:
                        print(f"⚠️ DB Fetch Error: {db_err}. Falling back to defaults.", flush=True)

                # Step 3: Format into array (Ensure the length exactly matches what X_scaler expects!)
                # Note: If your scaler was trained on 8 features including 'Extra', add it here.
                patient_metadata = np.array([[age, sex, height, weight, inf1, inf2, pace]])
                X_input_scaled = X_scaler.transform(patient_metadata)
                
                # --- C. PREDICT ---
                pred = model.predict([X_input_scaled, Y_input_scaled], verbose=0)[0]
                
                result = dict(zip(CLASSES, [float(p) for p in pred]))
                best_diagnosis = max(result, key=result.get)
                confidence = float(result[best_diagnosis])
                print(f"My predictions are: {result}", flush=True)
                
                diagnosis_names = {
                    'NORM': 'Normal Sinus Rhythm', 
                    'MI': 'Myocardial Infarction', 
                    'STTC': 'ST/T Change', 
                    'CD': 'Conduction Disturbance', 
                    'HYP': 'Hypertrophy'
                }
                readable_diagnosis = diagnosis_names.get(best_diagnosis, best_diagnosis)

                return jsonify({
                    "status": "success",
                    "diagnosis": readable_diagnosis,
                    "confidence": confidence,
                    "all_probabilities": result
                }), 200
            else:
                return jsonify({"error": "AI Model or Scalers are offline."}), 500

        except Exception as e:
            print(f"WFDB Processing Error: {e}", flush=True)
            return jsonify({"error": str(e)}), 500
# Start Server
# ==========================================
# --------- WEBSOCKET ECG ROUTES -----------
# ==========================================
@socketio.on('connect')
def handle_connect():
    print("✓ Client Connected to WebSockets (ESP32 or React Browser)")

# Create an empty dictionary at the top of your file to act as our RAM Buffer
active_ecg_buffers = {}

@socketio.on('esp32_ecg_stream')
def handle_ecg_stream(data):
    user_id = data.get('userId', 'guest')
    
    # 1. Extract the single voltage reading
    try:
        voltage = float(data.get('voltage', 0))
    except ValueError:
        return # Ignore garbage data

    # 2. Bounce the live point directly to React for the live animation
    emit('react_live_ecg', {"voltage": voltage}, broadcast=True)

    # 3. SHORT-TERM STORAGE: Add to the user's RAM buffer
    if user_id not in active_ecg_buffers:
        active_ecg_buffers[user_id] = []
        
    active_ecg_buffers[user_id].append(voltage)

    # 4. THE TRIGGER: Do we have exactly 5000 samples (10 seconds @ 500Hz)?
    TARGET_LENGTH = 5000
    
    if len(active_ecg_buffers[user_id]) >= TARGET_LENGTH:
        # Grab the full 10-second array
        full_10s_ecg_data = active_ecg_buffers[user_id][:TARGET_LENGTH]
        
        # Instantly empty the buffer so the next 10 seconds can start collecting immediately
        active_ecg_buffers[user_id] = [] 

        real_user_id = user_id # Fallback in case it's a guest or React test
        
        if 'db' in globals() and db is not None and user_id != 'guest':
            # Check if the incoming ID matches an RFID card in the system
            user_lookup = db.users.find_one({"card_id": user_id})
            if user_lookup:
                real_user_id = str(user_lookup["_id"])
                print(f"🔄 Link Found: Translated RFID [{user_id}] to DB ID [{real_user_id}]", flush=True)

        # Only print the first 5 numbers so you don't freeze your terminal!
        print(f"🔍 Data Preview (First 5 points): {full_10s_ecg_data[:5]}", flush=True)
        print(f"📦 {TARGET_LENGTH} samples collected for {user_id}! Running AI Prediction...", flush=True)

        # ---------------------------------------------------------
        # AI PREDICTION BLOCK
        # ---------------------------------------------------------
        # Default fallback variables in case the AI fails or is offline
        final_diagnosis = "Unknown"
        final_confidence = 0.0
        percentages_result = {}

        if model is not None and X_scaler is not None:
            try:
                # 1. Convert to Numpy Array
                raw_adc = np.array(full_10s_ecg_data, dtype=float)
                
                # 2. Convert to Volts (assuming 3.3V ESP32 logic)
                voltage = (raw_adc / 4095.0) * 3.3
                
                # 3. Center the signal at 0.0 (Remove DC offset)
                centered_voltage = voltage - np.mean(voltage)
                
                # 4. Convert to true Millivolts (Assuming AD8232 1100x Gain)
                ecg_mv = centered_voltage / 1.1
                
                # ---> 4.5 APPLY DIGITAL FILTER <---
                clean_ecg_mv = apply_bandpass_filter(ecg_mv, lowcut=0.5, highcut=40.0, fs=500.0)
                
                # 5. Prepare Signal (Y) -> Shape: (1, 5000, 1)
                Y_input = np.reshape(clean_ecg_mv, (1, TARGET_LENGTH, 1))
                
                # 6. APPLY THE .PKL SCALER 
                Y_input_scaled = (Y_input - Y_mean) / Y_std 
                
                # ---------------------------------------------------------
                # 7. Prepare Metadata (X) FROM MONGODB
                # ---------------------------------------------------------
                # Set default fallback values
                age, sex, height, weight, inf1, inf2, pace = 50.0, 1.0, 170.0, 70.0, 0.0, 0.0, 0.0
                
                # Try to fetch the real data from MongoDB
                if 'db' in globals() and db is not None and user_id != 'guest':
                    try:
                        user_data = db.medical_history.find_one({"user_id": str(user_id)})
                        if user_data:
                            # Safely extract and cast data from DB, using defaults if fields are empty
                            age = float(user_data.get('age') or age)
                            sex_val = user_data.get('sex', 'male')
                            sex = 1.0 if str(sex_val).lower() == 'male' else 0.0
                            height = float(user_data.get('height') or height)
                            weight = float(user_data.get('weight') or weight)
                            inf1 = float(user_data.get('inf1') or inf1)
                            inf2 = float(user_data.get('inf2') or inf2)
                            pace = float(user_data.get('pace') or pace)
                            
                            print(f"📊 Loaded DB Profile for Live Stream: User {user_id}", flush=True)
                        else:
                            print(f"⚠️ No DB profile found for user {user_id}, using defaults.", flush=True)
                    except Exception as db_err:
                        print(f"⚠️ DB Fetch Error: {db_err}. Falling back to defaults.", flush=True)

                # Format into array and Scale
                patient_metadata = np.array([[age, sex, height, weight, inf1, inf2, pace]])
                X_input_scaled = X_scaler.transform(patient_metadata)
                
                # ---------------------------------------------------------
                # 8. Predict
                # ---------------------------------------------------------
                pred = model.predict([X_input_scaled, Y_input_scaled], verbose=0)[0]
                
                # Map acronyms to readable names
                diagnosis_names = {
                    'NORM': 'Normal Sinus Rhythm', 
                    'MI': 'Myocardial Infarction', 
                    'STTC': 'ST/T Change', 
                    'CD': 'Conduction Disturbance', 
                    'HYP': 'Hypertrophy'
                }
                
                raw_result = dict(zip(CLASSES, [float(p) for p in pred]))
                best_diagnosis_code = max(raw_result, key=raw_result.get)
                
                # Convert ALL 5 probabilities into clean percentages
                for code, prob in raw_result.items():
                    readable_name = diagnosis_names.get(code, code)
                    percentages_result[readable_name] = round(prob * 100, 2)
                    
                # Set the final variables that will be emitted and saved
                final_diagnosis = diagnosis_names.get(best_diagnosis_code, best_diagnosis_code)
                final_confidence = percentages_result[final_diagnosis]

            except Exception as e:
                print(f"⚠️ AI Prediction Error: {e}", flush=True)
                final_diagnosis = "AI Error"
        else:
            final_diagnosis = "Model Offline"

        # ---------------------------------------------------------
        # EMIT RESULT TO FRONTEND (HAPPENS ONCE)
        # ---------------------------------------------------------
        emit('prediction_result', {
            "diagnosis": final_diagnosis, 
            "confidence": final_confidence,
            "all_probabilities": percentages_result
        }, broadcast=True)
        emit('esp32_prediction_feedback', {
            "diag": final_diagnosis,
            "code": best_diagnosis_code, # e.g., 'NORM', 'MI'
            "conf": final_confidence
        }, broadcast=True)
        # ---------------------------------------------------------
        # LONG-TERM STORAGE: Save to MongoDB
        # ---------------------------------------------------------
        if 'db' in globals() and db is not None:
            try:
                record_doc = {
                    "user_id": real_user_id,
                    "RF_ID": user_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "diagnosis": final_diagnosis,
                    "confidence": final_confidence,
                    "ecg_data_array": full_10s_ecg_data # Saves all 5000 raw points
                }
                db.ecg_records.insert_one(record_doc)
                print(f"💾 Record successfully saved to MongoDB for {real_user_id} with RFID of {user_id}", flush=True)
                
            except Exception as e:
                print(f"❌ Failed to save to database: {e}", flush=True)
        else:
            print("⚠️ Skipping DB save: MongoDB is not connected to the server.", flush=True)


@socketio.on('hardware_login_attempt')
def handle_hardware_login(data):
    card_id = data.get('cardId')
    try:
        user = db.users.find_one({"card_id": card_id})
        if not user:
            print(f"❌ DENIED: Unrecognized card tapped ({card_id})")
            emit('login_error', {"error": "Unrecognized card."}, broadcast=True)
            return
        print(f" ACCESSED:card tapped ({card_id})")
        # ... (your existing token generation code) ...
        if isinstance(token, bytes): token = token.decode('utf-8')

        # ==========================================
        # NEW: PRINT SUCCESS MESSAGE IN TERMINAL!
        # ==========================================
        print(f"\n✅ SUCCESS: Card {card_id} accepted!", flush=True)
        print(f"🏥 Logged in Doctor: Dr. {user.get('first_name')} {user.get('last_name')}\n", flush=True)

        emit('login_success', {
            "token": token,
            "user": {"id": str(user['_id']), "firstName": user.get('first_name', '')}
        }, broadcast=True)
        
    except Exception as e:
        emit('login_error', {"error": str(e)}, broadcast=True)      

# ==========================================
# --------- RESPIRATORY API ROUTES ---------
# ==========================================
@app.route('/api/analyze-spirometry', methods=['POST'])
def analyze_spirometry():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    data = request.json
    user_id = str(data.get('userId', 'guest'))
    
    try:
        fev1 = float(data.get('fev1', 0))
        fvc = float(data.get('fvc', 0))
        pef = float(data.get('pef', 0) or 0)
        
        # Rule-based diagnostic logic
        ratio = (fev1 / fvc) * 100 if fvc > 0 else 0
        ratio = round(ratio, 2)
        
        status = "success"
        if ratio < 70:
            diagnosis = "Possible Obstructive Defect (e.g., Asthma, COPD)"
            status = "warning"
            if fev1 < 1.5:
                status = "danger"
        elif fvc < 3.0: # Simplistic threshold for demonstration
            diagnosis = "Possible Restrictive Defect"
            status = "warning"
        else:
            diagnosis = "Normal Spirometry"
            
        result = {
            "status": status,
            "diagnosis": diagnosis,
            "ratio": ratio
        }
        
        # Save to DB if not guest
        if user_id != 'guest':
            db.respiratory_history.insert_one({
                "user_id": user_id,
                "fev1": fev1,
                "fvc": fvc,
                "pef": pef,
                "ratio": ratio,
                "diagnosis": diagnosis,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })
            print(f"💾 Spirometry saved for user {user_id}", flush=True)
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Spirometry Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

# ==========================================
# --------- NEUROLOGY API ROUTES -----------
# ==========================================
@app.route('/api/analyze-cognitive', methods=['POST'])
def analyze_cognitive():
    if db is None: return jsonify({"error": "Database unavailable"}), 503
    data = request.json
    user_id = str(data.get('userId', 'guest'))
    
    try:
        score = int(data.get('mocaScore', 0))
        recall = data.get('memoryRecall', True)
        
        status = "success"
        recommendation = ""
        
        if score >= 26:
            diagnosis = "Normal Cognitive Function"
        else:
            status = "warning"
            if score >= 18:
                diagnosis = "Mild Cognitive Impairment (MCI)"
                recommendation = "Consider follow-up and monitoring."
            elif score >= 10:
                diagnosis = "Moderate Cognitive Impairment"
                status = "danger"
                recommendation = "Neurological consult recommended."
            else:
                diagnosis = "Severe Cognitive Impairment"
                status = "danger"
                recommendation = "Immediate neurological consult and care planning needed."
                
        if not recall and score >= 26:
            diagnosis += " (Note: Delayed Recall missed)"
            recommendation = "Monitor short-term memory over time."
            status = "warning"
            
        result = {
            "status": status,
            "diagnosis": diagnosis,
            "score": score,
            "recommendation": recommendation
        }
        
        # Save to DB if not guest
        if user_id != 'guest':
            db.neurology_history.insert_one({
                "user_id": user_id,
                "moca_score": score,
                "memory_recall": recall,
                "diagnosis": diagnosis,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })
            print(f"💾 Cognitive test saved for user {user_id}", flush=True)
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Cognitive Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

# ==========================================
# --------- PORTAL FEATURE ROUTES ----------
# ==========================================

def to_base36(n):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    res = ""
    while n > 0:
        n, r = divmod(n, 36)
        res = chars[r] + res
    return res or "0"

@socketio.on('book_appointment')
def handle_book_appointment(data):
    if 'db' in globals() and db is not None:
        try:
            doctor = data.get('doctor')
            date = data.get('date')
            time_slot = data.get('time')

            # --- 1. CONFLICT DETECTION (Mutual Exclusion) ---
            existing = db.appointments.find_one({
                "doctor": doctor,
                "date": date,
                "time": time_slot,
                "status": {"$ne": "cancelled"}
            })

            if existing:
                # Find available slots for suggestions
                all_slots = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", 
                             "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00"]
                booked = db.appointments.find({
                    "doctor": doctor,
                    "date": date,
                    "status": {"$ne": "cancelled"}
                }).distinct("time")
                
                available = [s for s in all_slots if s not in booked]
                
                emit('appointment_error', {
                    "error": "This time slot is already booked for this doctor.",
                    "suggested": available[:5] # Suggest first 5 available
                }, broadcast=False)
                return

            # --- 2. Sequential Token Logic ---
            counter = db.counters.find_one_and_update(
                {"_id": "appointment_token"},
                {"$inc": {"sequence_value": 1}},
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
            seq = counter.get("sequence_value", 1)
            # Encode YYMMDDHHMM into Base36 for brevity
            now_val = int(datetime.datetime.now().strftime("%y%m%d%H%M"))
            encoded_date = to_base36(now_val)
            token_number = f"TKN-{encoded_date}-{seq:04d}"
            
            # Auto-resolve patient name
            user_id = data.get('userId')
            patient_name = data.get('patientName')
            if user_id != 'guest' and 'db' in globals() and db is not None:
                from bson.objectid import ObjectId
                try:
                    user = db.users.find_one({"_id": ObjectId(user_id)})
                except:
                    user = db.users.find_one({"_id": user_id})
                if user and user.get('role') != 'admin':
                    patient_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
            if not patient_name:
                patient_name = 'Unknown Patient'

            appointment_doc = {
                "user_id": user_id,
                "patient_name": patient_name,
                "department": data.get('department'),
                "doctor": data.get('doctor'),
                "date": data.get('date'),
                "time": data.get('time'),
                "status": "confirmed",
                "token_number": token_number,
                "created_at": datetime.datetime.now(datetime.timezone.utc)
            }
            db.appointments.insert_one(appointment_doc)
            emit('appointment_booked', {"status": "success", "message": "Appointment Confirmed", "token": token_number}, broadcast=False)
            
            user_apps = list(db.appointments.find({"user_id": data.get('userId')}, {"_id": 0}))
            for i, app_ in enumerate(user_apps):
                if "id" not in app_:
                    app_["id"] = app_.get("token_number", i + 100)
                if 'created_at' in app_ and app_['created_at']:
                    app_['created_at'] = app_['created_at'].isoformat()
            emit('appointment_update', user_apps, broadcast=False)
        except Exception as e:
            print(f"❌ Appointment Booking Error: {e}", flush=True)
            emit('appointment_error', {"error": str(e)}, broadcast=False)

@socketio.on('cancel_appointment')
def handle_cancel_appointment(data):
    if 'db' in globals() and db is not None:
        try:
            db.appointments.update_one({"token_number": data.get('id')}, {"$set": {"status": "cancelled"}})
        except Exception as e:
            pass

@socketio.on('delete_appointment')
def handle_delete_appointment(data):
    if 'db' in globals() and db is not None:
        try:
            db.appointments.delete_one({"token_number": data.get('id')})
        except Exception as e:
            pass

@socketio.on('get_appointments')
def handle_get_appointments(data):
    if 'db' in globals() and db is not None:
        try:
            user_id = data.get('userId')
            role = data.get('role', 'patient')
            
            # Verify role from DB
            from bson.objectid import ObjectId
            try:
                user = db.users.find_one({"_id": ObjectId(user_id)})
            except:
                user = db.users.find_one({"_id": user_id})
                
            if user and user.get('role') == 'admin':
                role = 'admin'

            if role == 'admin':
                # Admins get ALL appointments
                apps = list(db.appointments.find({}, {"_id": 0}))
            else:
                apps = list(db.appointments.find({"user_id": user_id}, {"_id": 0}))
                
            for i, app_ in enumerate(apps):
                if "id" not in app_:
                    app_["id"] = app_.get("token_number", i + 100)
                # Convert datetime objects to strings
                if 'created_at' in app_ and app_['created_at']:
                    app_['created_at'] = app_['created_at'].isoformat()
            emit('appointment_update', apps, broadcast=False)
        except Exception as e:
            print(f"❌ Get Appointments Error: {e}", flush=True)
            emit('appointment_error', {"error": str(e)}, broadcast=False)

# --- LAB REPORTS (HTTP) ---
@app.route('/api/reports/<string:user_id>', methods=['GET'])
def get_lab_reports(user_id):
    mock_reports = [
        {"date": "2026-04-15", "testName": "Complete Blood Count", "doctor": "Dr. Smith", "status": "Ready", "downloadUrl": "#"},
        {"date": "2026-04-22", "testName": "Chest X-Ray", "doctor": "Dr. Lee", "status": "In Progress", "downloadUrl": "#"}
    ]
    return jsonify({"status": "success", "reports": mock_reports}), 200

# --- DOCUMENTS (HTTP) ---
from flask import send_from_directory

@app.route('/api/upload_doc', methods=['POST'])
def upload_doc():
    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['document']
    user_id = request.form.get('userId', 'guest')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        if 'db' in globals() and db is not None and user_id != 'guest':
            db.user_documents.insert_one({
                "user_id": user_id,
                "original_filename": filename,
                "saved_filename": unique_filename,
                "type": "image" if filename.lower().endswith(('.png', '.jpg', '.jpeg')) else "pdf",
                "uploaded_at": datetime.datetime.now(datetime.timezone.utc)
            })
            
        return jsonify({"status": "success", "message": "File uploaded successfully"}), 200

@app.route('/api/documents/<string:user_id>', methods=['GET'])
def get_documents(user_id):
    if 'db' in globals() and db is not None:
        try:
            docs = list(db.user_documents.find({"user_id": user_id}, {"_id": 0}).sort("uploaded_at", -1))
            formatted_docs = []
            for d in docs:
                file_url = f"{request.host_url}uploads/{d['saved_filename']}"
                formatted_docs.append({
                    "filename": d['original_filename'],
                    "type": d.get('type', 'pdf'),
                    "date": d['uploaded_at'].strftime("%Y-%m-%d") if 'uploaded_at' in d else "Unknown",
                    "url": file_url
                })
            return jsonify({"status": "success", "documents": formatted_docs}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"status": "success", "documents": []}), 200

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Start Server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"SocketIO Server running on port {port}")
    # Change app.run to socketio.run!
    socketio.run(app, host='0.0.0.0', port=port)