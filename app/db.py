from pymongo import MongoClient
import datetime

MONGODB_URI = "mongodb+srv://GoPredict:5vvgj23hbz@cluster0.uxpju.mongodb.net/Chatbot?retryWrites=true&w=majority"
client = MongoClient(MONGODB_URI)

db = client.Chatbot
messages = db["messages"]
sessions = db["sessions"]

def serialize_doc(doc):
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def save_message(doc: dict):
    """Insert one message document into messages collection"""
    messages.insert_one(doc)

    session_id = doc["session_id"]
    case_id = doc["case_id"]
    sessions.update_one(
        {"session_id": session_id},
        {"$set": {"user_id": doc["user_id"],
                  "org_id": doc["org_id"],
                  "channel": doc["channel"],
                  "last_updated": datetime.datetime.utcnow()},
         "$addToSet": {"case_ids": case_id}},  # prevent duplicates
        upsert=True
    )

def get_case_memory(user_id=None, session_id=None):
    """Fetch past messages for a given user or session, sorted by timestamp"""
    query = {}
    if user_id:
        query["user_id"] = user_id
    if session_id:
        query["session_id"] = session_id
    docs = list(messages.find(query).sort("timestamp", 1))
    return [serialize_doc(d) for d in docs]

def get_session_case_ids(session_id):
    """Fetch list of case_ids for a session"""
    doc = sessions.find_one({"session_id": session_id})
    return doc.get("case_ids", []) if doc else []

def get_messages_by_case_ids(case_ids):
    """Fetch messages by case_ids sorted by timestamp"""
    docs = list(messages.find({"case_id": {"$in": case_ids}}).sort("timestamp", 1))
    return [serialize_doc(d) for d in docs]
