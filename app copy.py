import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import pooling
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pickle
import wfdb
import uuid

# Auth controller
from auth_controller import signup, login

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) 

# --- Database Connection Pool ---
dbconfig = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "@"),
    "database": "hospital_automanager"
}
pool = mysql.connector.pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **dbconfig)

# --- Directory Configuration ---
UPLOAD_FOLDER = 'uploads/'
TEMP_FOLDER = 'temp/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True) # Used for WFDB processing
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MACHINE LEARNING MODEL ---
try:
    # Load the model
    model = tf.keras.models.load_model("ecg_model.h5")
    print("✓ Deep Learning Model loaded successfully!")
    
    # Load the dynamic classes from the pickle file
    with open("ecg_classes.pkl", "rb") as f:
        CLASSES = pickle.load(f)
    print(f"✓ Classes loaded from pickle: {CLASSES}")
    
except Exception as e:
    print("⚠️ Warning: Could not load model or classes. Error:", e)
    model = None
    # Fallback classes
    CLASSES = ['CD','HYP','MI','NORM','STTC']


# ==========================================
# -------------- AUTH ROUTES ---------------
# ==========================================
@app.route('/api/auth/signup', methods=['POST'])
def signup_route():
    return signup(request, pool)

@app.route('/api/auth/login', methods=['POST'])
def login_route():
    return login(request, pool)


# ==========================================
# --------- MEDICAL HISTORY ROUTES ---------
# ==========================================
@app.route('/api/medical-history/<int:user_id>', methods=['GET'])
def get_medical_history(user_id):
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True) 
        
        sql = "SELECT * FROM medical_history WHERE user_id = %s"
        cursor.execute(sql, (user_id,))
        result = cursor.fetchall()
        
        if result:
            return jsonify(result[0])
        else:
            return jsonify(None)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

@app.route('/api/medical-history', methods=['POST'])
def save_medical_history():
    data = request.json
    user_id = data.get('userId')
    age = data.get('age')
    sex = data.get('sex')
    family_history = 1 if data.get('familyHistory') == 'yes' else 0
    past_heart_problem = data.get('pastHeartProblem')

    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO medical_history (user_id, age, sex, family_heart_history, past_heart_problem) 
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            age = VALUES(age), 
            sex = VALUES(sex), 
            family_heart_history = VALUES(family_heart_history), 
            past_heart_problem = VALUES(past_heart_problem)
        """
        cursor.execute(sql, (user_id, age, sex, family_history, past_heart_problem))
        conn.commit()
        
        return jsonify({"message": "Medical history saved successfully!"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


# ==========================================
# --------- LIFESTYLE DATA ROUTES ----------
# ==========================================
@app.route('/api/lifestyle-data/<int:user_id>', methods=['GET'])
def get_lifestyle_data(user_id):
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        sql = "SELECT * FROM lifestyle_data WHERE user_id = %s"
        cursor.execute(sql, (user_id,))
        result = cursor.fetchall()
        
        if result:
            db_data = result[0]
            react_data = {
                "physicalActivity": db_data.get("physical_activity", ""),
                "smoking": db_data.get("smoking", ""),
                "alcoholUse": db_data.get("alcohol_use", ""),
                "otherSubstances": db_data.get("other_substances", "")
            }
            return jsonify(react_data)
        else:
            return jsonify(None)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

@app.route('/api/lifestyle-data', methods=['POST'])
def save_lifestyle_data():
    data = request.json
    user_id = data.get('userId')
    physical_activity = data.get('physicalActivity')
    smoking = data.get('smoking')
    alcohol_use = data.get('alcoholUse')
    other_substances = data.get('otherSubstances')

    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO lifestyle_data (user_id, physical_activity, smoking, alcohol_use, other_substances) 
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            physical_activity = VALUES(physical_activity), 
            smoking = VALUES(smoking), 
            alcohol_use = VALUES(alcohol_use), 
            other_substances = VALUES(other_substances)
        """
        cursor.execute(sql, (user_id, physical_activity, smoking, alcohol_use, other_substances))
        conn.commit()
        
        return jsonify({"message": "Lifestyle data saved successfully!"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


# ==========================================
# --------- FILE UPLOAD ROUTE --------------
# ==========================================
@app.route('/api/upload-report', methods=['POST'])
def upload_report():
    if 'report' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['report']
    user_id = request.form.get('userId')
    report_type = request.form.get('reportType')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        timestamp = str(int(time.time() * 1000))
        filename = timestamp + '-' + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(file_path)

        try:
            conn = pool.get_connection()
            cursor = conn.cursor()
            sql = "INSERT INTO health_reports (user_id, report_type, file_path) VALUES (%s, %s, %s)"
            cursor.execute(sql, (user_id, report_type, file_path))
            conn.commit()
            
            return jsonify({
                "message": "File uploaded and recorded in database!", 
                "filePath": file_path
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals(): conn.close()


# ==========================================
# ------- ML LIVE PREDICTION ROUTE ---------
# ==========================================
@app.route('/api/analyze-live-ecg', methods=['POST'])
def analyze_live_ecg():
    data = request.json
    user_id = data.get('userId')
    raw_data = data.get('ecgData')

    if not raw_data or len(raw_data) == 0:
        return jsonify({"error": "No ECG data received"}), 400

    if model is None:
        return jsonify({"error": "Machine Learning model is not loaded on the server"}), 500

    try:
        # 1. Convert to NumPy array
        signal = np.array(raw_data, dtype=float)
        
        # 2. Fix the Length (Model expects exactly 1000 points)
        if len(signal) > 1000:
            signal = signal[:1000] # Crop if too long
        else:
            # Pad with zeros if too short
            signal = np.pad(signal, (0, 1000 - len(signal)), 'constant')
            
        # 3. Fix the Leads (AD8232 is 1 lead, Model expects 3 leads)
        final_signal = np.zeros((1000, 3))
        final_signal[:, 0] = signal
        
        # 4. Normalize the data
        X = np.expand_dims(final_signal, 0)
        X = (X - X.mean()) / (X.std() + 1e-8) 
        
        # 5. Predict using TensorFlow!
        pred = model.predict(X)[0]
        
        result = dict(zip(CLASSES, [float(p) for p in pred]))
        best_diagnosis_code = max(result, key=result.get)
        
        # Map abbreviations to real names
        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }
        
        readable_diagnosis = diagnosis_names.get(best_diagnosis_code, best_diagnosis_code)

        return jsonify({
            "message": "Success",
            "diagnosis": readable_diagnosis,
            "confidence": result[best_diagnosis_code],
            "all_probabilities": result
        }), 200

    except Exception as e:
        print("ML Processing Error:", str(e))
        return jsonify({"error": "Failed to process ECG data", "details": str(e)}), 500


# ==========================================
# ------- WFDB (.dat & .hea) PREDICTION ----
# ==========================================
@app.route('/api/analyze-wfdb', methods=['POST'])
def analyze_wfdb():
    if 'dat' not in request.files or 'hea' not in request.files:
        return jsonify({"error": "Please upload BOTH the .dat and .hea files"}), 400

    dat_file = request.files['dat']
    hea_file = request.files['hea']

    if model is None:
        return jsonify({"error": "Machine Learning model is not loaded"}), 500

    # Give them a matching unique ID so wfdb can read them together
    uid = str(uuid.uuid4())
    dat_path = os.path.join(TEMP_FOLDER, f"{uid}.dat")
    hea_path = os.path.join(TEMP_FOLDER, f"{uid}.hea")

    try:
        # Save the files temporarily
        dat_file.save(dat_path)
        hea_file.save(hea_path)
        
        # Base path for wfdb (without the extension)
        base_path = os.path.join(TEMP_FOLDER, uid)

        # 1. Read the WFDB record
        signal, fields = wfdb.rdsamp(base_path)

        # 2. Format the data (Crop/Pad to exactly 1000 points)
        if len(signal) > 1000:
            signal = signal[:1000]
        else:
            signal = np.pad(signal, ((0, 1000 - len(signal)), (0, 0)), 'constant')

        # 3. Ensure exactly 3 leads (Model expects 1000, 3)
        final_signal = np.zeros((1000, 3))
        leads_to_copy = min(signal.shape[1], 3)
        final_signal[:, :leads_to_copy] = signal[:, :leads_to_copy]

        # 4. Normalize and Predict
        X = np.expand_dims(final_signal, 0)
        X = (X - X.mean()) / (X.std() + 1e-8) 

        pred = model.predict(X)[0]

        # 5. Get the result
        result = dict(zip(CLASSES, [float(p) for p in pred]))
        best_diagnosis_code = max(result, key=result.get)

        diagnosis_names = {
            'NORM': 'Normal Sinus Rhythm',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }

        readable_diagnosis = diagnosis_names.get(best_diagnosis_code, best_diagnosis_code)

        # Clean up the temporary files
        os.remove(dat_path)
        os.remove(hea_path)

        return jsonify({
            "message": "Success",
            "diagnosis": readable_diagnosis,
            "confidence": result[best_diagnosis_code],
            "sampling_rate": fields.get('fs', 'Unknown')
        }), 200

    except Exception as e:
        print("WFDB Processing Error:", str(e))
        # Clean up files if an error occurs
        if os.path.exists(dat_path): os.remove(dat_path)
        if os.path.exists(hea_path): os.remove(hea_path)
        return jsonify({"error": "Failed to process ECG files. Ensure they are valid WFDB format.", "details": str(e)}), 500


# Start Server
if __name__ == '__main__':
    print("Flask Server running on http://localhost:5000")
    app.run(port=5000, debug=True)