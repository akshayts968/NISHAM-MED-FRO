import os
import bcrypt
import jwt
import datetime
from flask import jsonify
from pymongo.errors import DuplicateKeyError

def signup(request, db):
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password_bytes = bcrypt.hashpw(password.encode('utf-8'), salt)
        hashed_password = hashed_password_bytes.decode('utf-8')

        # Create the user document
        user_doc = {
            "first_name": data.get('firstName'),
            "last_name": data.get('lastName'),
            "email": email,
            "mobile": data.get('mobile'),
            "password": hashed_password,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        }

        # Insert into MongoDB
        db.users.insert_one(user_doc)
        return jsonify({"message": "User registered successfully!"}), 201

    except DuplicateKeyError:
        return jsonify({"error": "An account with this email already exists"}), 409
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


def login(request, db):
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        # Find user by email
        user = db.users.find_one({"email": email})

        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        # Compare passwords
        stored_password = user['password'].encode('utf-8')
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password):
            return jsonify({"error": "Invalid email or password"}), 401

        secret_key = os.environ.get('JWT_SECRET', 'fallback_secret_for_dev_only')
        
        payload = {
            'id': str(user['_id']), # Convert MongoDB ObjectId to string
            'email': user['email'],
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
        }
        
        token = jwt.encode(payload, secret_key, algorithm='HS256')
        if isinstance(token, bytes):
            token = token.decode('utf-8')

        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": str(user['_id']), 
                "firstName": user.get('first_name'), 
                "lastName": user.get('last_name')
            }
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Authentication failed", "details": str(e)}), 500