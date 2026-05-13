import redis
import json

SESSION_PREFIX = "session:"

try:
    redis_client = redis.StrictRedis(
        host='redis-10058.c326.us-east-1-3.ec2.cloud.redislabs.com',
        port=10058,
        decode_responses=False,
        username="default",
        password="LuPiDTpjW6rXWNcqxcl0A2V0eMK9YAzj",
        db=0,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("[OK] Redis: connected")
except Exception as e:
    redis_client = None
    REDIS_AVAILABLE = False
    print(f"[WARN] Redis unavailable: {e}. In-memory fallback active.")


def load_redis_memory(user_id, session_id):
    if not REDIS_AVAILABLE or redis_client is None:
        return []
    try:
        key = f"{user_id}:{session_id}"
        data = redis_client.get(key)
        if data:
            memory_data = json.loads(data)
            if not isinstance(memory_data, list):
                return []
            return memory_data
    except Exception as e:
        print(f"[WARN] Redis load failed: {e}")
    return []


def save_redis_memory(user_id, session_id, memory):
    if not REDIS_AVAILABLE or redis_client is None:
        return
    try:
        key = f"{user_id}:{session_id}"
        redis_client.set(key, json.dumps(memory.memory))
        redis_client.expire(key, 3600)
    except Exception as e:
        print(f"[WARN] Redis save failed: {e}")


def clear_redis_memory(session_id):
    if not REDIS_AVAILABLE or redis_client is None:
        return
    try:
        keys = redis_client.keys(f"*:{session_id}")
        for key in keys:
            redis_client.delete(key)
    except Exception as e:
        print(f"[WARN] Redis clear failed: {e}")
