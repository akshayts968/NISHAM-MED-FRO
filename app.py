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
from pymongo import MongoClient
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

# Use an 'r' before the string so Windows backslashes don't break the path
MODEL_PATH = r"D:\Nisham med\meb\model_1lead_500hz.h5"
X_SCALER_PATH = r"D:\Nisham med\meb\X_scaler_500hz.pkl"
Y_SCALER_PATH = r"D:\Nisham med\meb\Y_scaler_stats_500hz.pkl"

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
        signal = np.array(raw_data, dtype=float)
        
        # 1. Crop or Pad to exactly 5000 points (10 seconds @ 500Hz)
        TARGET_LENGTH = 5000
        if len(signal) > TARGET_LENGTH:
            signal = signal[:TARGET_LENGTH] 
        elif len(signal) < TARGET_LENGTH:
            signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)), 'constant')
            
        # 2. Prepare the Signal (Y) -> Shape: (1, 5000, 1)
        # We no longer pad it to 3 columns. It is strictly 1-lead.
        Y_input = np.reshape(signal, (1, TARGET_LENGTH, 1))
        
        # Scale using your pre-trained global stats from the .pkl file
        Y_input_scaled = (Y_input - Y_mean) / Y_std 
        
        # 3. Prepare the Metadata (X) -> Shape: (1, 7)
        # Exactly 7 features: [Age=50, Sex=1(Male), Height=170, Weight=70, Inf1=0, Inf2=0, Pace=0]
        dummy_metadata = np.array([[50, 1, 170, 70, 0, 0, 0]])
        X_input_scaled = X_scaler.transform(dummy_metadata)
        
        # 4. Normalize and Predict (Pass BOTH inputs to the multi-modal model)
        pred = model.predict([X_input_scaled, Y_input_scaled], verbose=0)[0]
        
        # 5. Format the Results
        result = dict(zip(CLASSES, [float(p) for p in pred]))
        best_diagnosis_code = max(result, key=result.get)
        
        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }
        
        # Convert output to percentages for easier frontend reading
        for key in result:
            result[key] = round(result[key] * 100, 2)
        
        return jsonify({
            "message": "Success",
            "diagnosis": diagnosis_names.get(best_diagnosis_code, best_diagnosis_code),
            "confidence": result[best_diagnosis_code],
            "all_probabilities": result
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
            
            # Extract the raw signal array (Assuming Lead II is channel 0 or 1 depending on the file)
            # You might need to adjust the index if Lead II isn't the first channel in your specific .dat files
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
                # Shape it to (1, 5000, 1) for the 1D CNN
                Y_input = np.reshape(signal_data, (1, TARGET_LENGTH, 1))
                
                # Scale using the saved training statistics (Globals loaded at app startup)
                Y_input_scaled = (Y_input - Y_mean) / Y_std 
                
                # --- B. PREPARE THE METADATA (X) ---
                # Since the WFDB upload doesn't have patient vitals, we use a default baseline profile.
                # [Age=50, Sex=1(Male), Height=170, Weight=70, Inf1=0, Inf2=0, Pace=0, Extra=0]
                dummy_metadata = np.array([[50, 1, 170, 70, 0, 0, 0]])
                X_input_scaled = X_scaler.transform(dummy_metadata)
                
                # --- C. PREDICT ---
                # The model requires both inputs in a list: [Metadata, Signal]
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
                    "all_probabilities": result # Helpful for debugging frontend
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
        
        # Only print the first 5 numbers so you don't freeze your terminal!
        print(f"🔍 Data Preview (First 5 points): {full_10s_ecg_data}", flush=True)
        print(f"📦 {TARGET_LENGTH} samples collected for {user_id}! Running AI Prediction...", flush=True)

        # ---------------------------------------------------------
        # AI PREDICTION BLOCK
        # ---------------------------------------------------------
        best_diagnosis = "Unknown"
        confidence = 0.0

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
                print("\n🔍 --- CONVERSION PREVIEW (First 5 points) ---", flush=True)
                print(f"1. Raw ADC      : {np.round(raw_adc[:5], 1)}")
                print(f"2. Volts (V)    : {np.round(voltage[:5], 3)}")
                print(f"3. Centered (V) : {np.round(centered_voltage[:5], 3)}")
                print(f"4. True ECG (mV): {np.round(ecg_mv[:5], 3)}")
                print("----------------------------------------------\n", flush=True)
                # 5. Prepare Signal (Y) -> Shape: (1, 5000, 1)
                Y_input = np.reshape(ecg_mv, (1, TARGET_LENGTH, 1))
                
                # ---> 6. APPLY THE .PKL SCALER (THE CRITICAL FIX) <---
                Y_input_scaled = (Y_input - Y_mean) / Y_std 
                
                # 7. Prepare Metadata (X) -> Shape: (1, 7)
                dummy_metadata = np.array([[50, 1, 170, 70, 0, 0, 0]])
                X_input_scaled = X_scaler.transform(dummy_metadata)
                
                # 8. Predict (Pass the scaled inputs!)
                pred = model.predict([X_input_scaled, Y_input_scaled], verbose=0)[0]
                result = dict(zip(CLASSES, [float(p) for p in pred]))
                
                best_diagnosis = max(result, key=result.get)
                confidence = round(result[best_diagnosis] * 100, 1)

            except Exception as e:
                print(f"⚠️ AI Prediction Error: {e}", flush=True)
                best_diagnosis = "AI Error"
        else:
            best_diagnosis = "Model Offline"

        # Map acronyms to readable names
        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm', 
            'MI': 'Myocardial Infarction', 
            'STTC': 'ST/T Change', 
            'CD': 'Conduction Disturbance', 
            'HYP': 'Hypertrophy'
        }
        readable_diagnosis = diagnosis_names.get(best_diagnosis, best_diagnosis)

        # Send the result to the React screen
        emit('prediction_result', {
            "diagnosis": readable_diagnosis, 
            "confidence": confidence
        }, broadcast=True)

        # ---------------------------------------------------------
        # LONG-TERM STORAGE: Save to MongoDB
        # ---------------------------------------------------------
        # Only attempt to save if the 'db' variable actually exists and is connected
        if 'db' in globals() and db is not None:
            try:
                record_doc = {
                    "user_id": user_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "diagnosis": readable_diagnosis,
                    "confidence": confidence,
                    "ecg_data_array": full_10s_ecg_data # Saves all 5000 points!
                }
                # Insert into a new 'ecg_records' collection in your database
                db.ecg_records.insert_one(record_doc)
                print(f"💾 Record successfully saved to MongoDB for {user_id}", flush=True)
                
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

# Start Server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"SocketIO Server running on port {port}")
    # Change app.run to socketio.run!
    socketio.run(app, host='0.0.0.0', port=port)