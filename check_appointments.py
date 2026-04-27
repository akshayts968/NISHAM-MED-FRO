import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client[os.environ.get("DB_DATABASE", "hospital_automanager")]

appointments = list(db.appointments.find())
print(f"Total appointments: {len(appointments)}")
for app in appointments:
    print(app)
