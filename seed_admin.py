import os
import bcrypt
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client[os.environ.get("DB_DATABASE", "hospital_automanager")]

admin_email = "admin@antigravity.med"
admin_password = "Admin123!"

existing = db.users.find_one({"email": admin_email})
if existing:
    print("Admin already exists!")
else:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(admin_password.encode('utf-8'), salt).decode('utf-8')

    admin_doc = {
        "first_name": "System",
        "last_name": "Administrator",
        "email": admin_email,
        "mobile": "0000000000",
        "password": hashed_password,
        "role": "admin",
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }

    db.users.insert_one(admin_doc)
    print("Admin successfully seeded in medical_db.users!")
