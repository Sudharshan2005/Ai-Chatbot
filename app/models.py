# models.py
from pymongo import MongoClient
from bson import ObjectId

client = MongoClient("mongodb://localhost:27017/")
db = client["support_chat"]
users_collection = db["users"]

class User:
    @staticmethod
    def find_by_id(user_id: str):
        return users_collection.find_one({"_id": user_id})

    @staticmethod
    def update(user_id: str, patch: dict):
        return users_collection.find_one_and_update(
            {"_id": user_id},
            {"$set": patch},
            return_document=True
        )