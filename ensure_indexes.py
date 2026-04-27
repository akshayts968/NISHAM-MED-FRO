import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client[os.environ.get("DB_DATABASE", "hospital_automanager")]

# Create a partial unique index: Only unique if status is NOT cancelled
print("Creating unique index for doctor appointments...")
db.appointments.create_index(
    [("doctor", ASCENDING), ("date", ASCENDING), ("time", ASCENDING)],
    unique=True,
    partialFilterExpression={"status": {"$ne": "cancelled"}}
)
print("✓ Unique index created successfully.")
