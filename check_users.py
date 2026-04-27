import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client[os.environ.get("DB_DATABASE", "hospital_automanager")]

users = list(db.users.find())
for u in users:
    print(f"Email: {u.get('email')}, Role: {u.get('role')}")
