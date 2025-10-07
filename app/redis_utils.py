import redis
import pickle
import datetime
from memory import CaseMemory
from db import get_case_memory   
import json

redis_client = redis.StrictRedis(
    host='redis-10914.c305.ap-south-1-1.ec2.redns.redis-cloud.com',
    port=10914,
    decode_responses=False,
    username="default",
    password="VJVvAaqi3v4ydNvyUcKkbWzrtUACryx2",
    db=0
)

SESSION_PREFIX = "session:"


# def load_redis_memory(user_id: str, session_id: str):
#     """Load memory from Redis first, fallback to DB if not found."""
#     cached_memory = redis_client.get(f"session:{session_id}")
#     if cached_memory:
#         return pickle.loads(cached_memory)

#     past_messages = get_case_memory(user_id=user_id, session_id=session_id)
#     initial_memory = [
#         {
#             "user": m["user_message"],
#             "bot": m["response"],
#             "intent": m.get("nlu", {}).get("intent", "other"),
#         }
#         for m in past_messages
#     ]
#     memory = CaseMemory(initial_memory=initial_memory)

#     redis_client.set(f"session:{session_id}", pickle.dumps(memory), ex=3600)
#     return memory


# def save_redis_memory(user_id, session_id, memory):
#     key = f"{user_id}:{session_id}"
#     redis_client.set(key, json.dumps(memory.memory))  # Serialize memory.memory
#     redis_client.expire(key, 3600)


# def clear_redis_memory(session_id: str):
#     """Clear memory manually (end of session)."""
#     redis_client.delete(f"session:{session_id}")

def load_redis_memory(user_id, session_id):
    key = f"{user_id}:{session_id}"
    data = redis_client.get(key)
    if data:
        try:
            memory_data = json.loads(data)
            if not isinstance(memory_data, list):
                print(f"Error: Redis data for {key} is not a list: {memory_data}")
                return []
            return memory_data
        except json.JSONDecodeError as e:
            print(f"Error decoding Redis memory for {key}: {e}")
            return []
    return []

def save_redis_memory(user_id, session_id, memory):
    key = f"{user_id}:{session_id}"
    try:
        redis_client.set(key, json.dumps(memory.memory))  # Serialize memory.memory
        redis_client.expire(key, 3600)  # Set TTL (1 hour)
        print(f"Saved memory to Redis for {key}: {memory.memory}")
    except Exception as e:
        print(f"Error saving Redis memory for {key}: {e}")

def clear_redis_memory(session_id):
    keys = redis_client.keys(f"*:{session_id}")
    for key in keys:
        redis_client.delete(key)
        print(f"Cleared Redis key: {key}")