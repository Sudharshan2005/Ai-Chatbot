import redis
import pickle
import datetime
from memory import CaseMemory
from db import get_case_memory   

redis_client = redis.StrictRedis(
    host='redis-10914.c305.ap-south-1-1.ec2.redns.redis-cloud.com',
    port=10914,
    decode_responses=False,
    username="default",
    password="VJVvAaqi3v4ydNvyUcKkbWzrtUACryx2",
    db=0
)

SESSION_PREFIX = "session:"


def load_redis_memory(user_id: str, session_id: str):
    """Load memory from Redis first, fallback to DB if not found."""
    cached_memory = redis_client.get(f"session:{session_id}")
    if cached_memory:
        return pickle.loads(cached_memory)

    past_messages = get_case_memory(user_id=user_id, session_id=session_id)
    initial_memory = [
        {
            "user": m["user_message"],
            "bot": m["response"],
            "intent": m.get("nlu", {}).get("intent", "other"),
        }
        for m in past_messages
    ]
    memory = CaseMemory(initial_memory=initial_memory)

    redis_client.set(f"session:{session_id}", pickle.dumps(memory), ex=3600)
    return memory


def save_redis_memory(session_id: str, memory):
    """Save memory to Redis with TTL (hybrid mode)."""
    redis_client.set(f"session:{session_id}", pickle.dumps(memory), ex=3600)


def clear_redis_memory(session_id: str):
    """Clear memory manually (end of session)."""
    redis_client.delete(f"session:{session_id}")