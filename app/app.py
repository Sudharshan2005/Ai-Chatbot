from flask import Flask, request, jsonify
import joblib, os, time, datetime, pickle
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from db import save_message, get_case_memory, get_session_case_ids, get_messages_by_case_ids, serialize_doc
from redis_utils import load_redis_memory, save_redis_memory, clear_redis_memory
from mq import setup_topology, publish_event
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS


load_dotenv()
df, X, embedding_model = joblib.load("/Users/sudharshan/Documents/PS-G538/chatbot.pkl")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
bot = genai.GenerativeModel(model_name='models/gemini-2.5-flash')

setup_topology()

class CaseMemory:
    def __init__(self, initial_memory=None):
        self.memory = initial_memory or []

    def add_interaction(self, user_message, bot_response, intent):
        self.memory.append({
            "user": user_message,
            "bot": bot_response,
            "intent": intent,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    def get_context(self):
        return "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in self.memory])

# sessions_cache = {}

def analyze_intent(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["hello", "hi", "hey"]): return "greeting"
    elif any(word in q for word in ["bye", "goodbye", "see you"]): return "farewell"
    elif "name" in q: return "name_query"
    elif "order" in q or "status" in q: return "order_status"
    elif "complaint" in q or "not working" in q: return "complaint"
    elif "help" in q or "how to" in q: return "faq_query"
    else: return "other"

def find_best_dataset_answer(query, threshold=0.85):
    query_vec = embedding_model.encode([query])
    sims = cosine_similarity(query_vec, X)[0]
    best_idx = sims.argmax()
    if sims[best_idx] >= threshold:
        return df.iloc[best_idx]["cleaned_answer"]
    return None

app = Flask(__name__)
app.config["SECRET_KEY"] = "test-secret"
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins="*"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "")
    session_id = data.get("session_id") or str(uuid4())
    user_id = data.get("user_id", "user_123")
    org_id = data.get("org_id", "acme")
    channel = data.get("channel", "web")
    request_id = data.get("request_id") or str(uuid4())
    case_id = str(uuid4())
    message_id = str(uuid4())
    t0 = time.perf_counter()

    memory = load_redis_memory(user_id, session_id)
    if not memory:
        case_ids = get_session_case_ids(session_id)
        past_messages = get_messages_by_case_ids(case_ids)
        initial_memory = [{"user": m["user_message"], "bot": m["response"], "intent": m.get("nlu", {}).get("intent","other")} 
                          for m in past_messages]
        memory = CaseMemory(initial_memory=initial_memory)

    # if session_id not in sessions_cache:
    #     case_ids = get_session_case_ids(session_id)
    #     past_messages = get_messages_by_case_ids(case_ids)
    #     initial_memory = [{"user": m["user_message"], "bot": m["response"], "intent": m.get("nlu", {}).get("intent","other")} 
    #                       for m in past_messages]
    #     sessions_cache[session_id] = CaseMemory(initial_memory=initial_memory)
    # memory = sessions_cache[session_id]

    intent = analyze_intent(user_message)
    intent_confidence = 0.7 if intent != "other" else 0.5

    for past in memory.memory:
        if past["user"].lower() == user_message.lower():
            latency_ms = round((time.perf_counter() - t0) * 1000)
            doc = {
                "org_id": org_id, "user_id": user_id, "channel": channel,
                "session_id": session_id, "case_id": case_id, "message_id": message_id,
                "parent_message_id": None, "request_id": request_id, "direction": "inbound",
                "user_message": user_message, "response": past["bot"], "source": "case_memory",
                "status": "resolved",
                "nlu": {"intent": intent, "intent_confidence": intent_confidence, "language": "en", "sentiment": "neutral", "tone": "neutral"},
                "retrieval": {"kb_id": "default", "top_k_doc_ids": [], "answer_confidence": 1.0, "similarity_score": 1.0},
                "llm": {"model": None, "latency_ms": latency_ms, "prompt_tokens": None, "completion_tokens": None},
                "ticket": {"escalated": False, "ticket_id": None, "resolution_code": None},
                "feedback": {"user_rating": None, "user_comment": None},
                "security": {"pii_redacted": True, "pii_types": []},
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            save_message(doc)
            save_redis_memory(session_id, memory)
            publish_event(
                body=serialize_doc(doc),
                headers={"type": "message.received", "x-attempt": 0}
            )
            socketio.emit("message:ack", serialize_doc(doc), room=session_id)
            return jsonify(serialize_doc(doc))

    ans = find_best_dataset_answer(user_message)
    if ans:
        sim = float(cosine_similarity(embedding_model.encode([user_message]), X)[0].max())
        memory.add_interaction(user_message, ans, intent)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        doc = {
            "org_id": org_id, "user_id": user_id, "channel": channel,
            "session_id": session_id, "case_id": case_id, "message_id": message_id,
            "parent_message_id": None, "request_id": request_id, "direction": "inbound",
            "user_message": user_message, "response": ans, "source": "dataset", "status": "resolved",
            "nlu": {"intent": intent, "intent_confidence": intent_confidence, "language": "en", "sentiment": "neutral", "tone": "neutral"},
            "retrieval": {"kb_id": "default", "top_k_doc_ids": [], "answer_confidence": sim, "similarity_score": sim},
            "llm": {"model": None, "latency_ms": latency_ms, "prompt_tokens": None, "completion_tokens": None},
            "ticket": {"escalated": False, "ticket_id": None, "resolution_code": None},
            "feedback": {"user_rating": None, "user_comment": None},
            "security": {"pii_redacted": True, "pii_types": []},
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        save_message(doc)
        save_redis_memory(session_id, memory)
        publish_event(
            body=serialize_doc(doc),
            headers={"type": "message.received", "x-attempt": 0}
        )
        socketio.emit("message:ack", serialize_doc(doc), room=session_id)
        return jsonify(serialize_doc(doc))

    ctx = memory.get_context()
    prompt = f"You are a helpful and friendly customer support chatbot. Your goal is to provide clear, concise, and accurate responses to user inquiries. Keep your answers short to medium in length. Your tone should be warm, empathetic, and non-judgmental. Here's the conversation so far: {ctx} New user message: {user_message} Craft a helpful response that addresses the user's message directly. Ensure your response is easy to understand and maintains a consistent, supportive tone."
    resp_text = bot.generate_content(prompt).text.strip()
    memory.add_interaction(user_message, resp_text, intent)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    doc = {
        "org_id": org_id, "user_id": user_id, "channel": channel,
        "session_id": session_id, "case_id": case_id, "message_id": message_id,
        "parent_message_id": None, "request_id": request_id, "direction": "inbound",
        "user_message": user_message, "response": resp_text, "source": "gemini", "status": "open",
        "nlu": {"intent": intent, "intent_confidence": intent_confidence, "language": "en", "sentiment": "neutral", "tone": "neutral"},
        "retrieval": {"kb_id": "default", "top_k_doc_ids": [], "answer_confidence": None, "similarity_score": None},
        "llm": {"model": "gemini-2.5-flash", "latency_ms": latency_ms, "prompt_tokens": None, "completion_tokens": None},
        "ticket": {"escalated": False, "ticket_id": None, "resolution_code": None},
        "feedback": {"user_rating": None, "user_comment": None},
        "security": {"pii_redacted": True, "pii_types": []},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    save_message(doc)
    save_redis_memory(session_id, memory)
    publish_event(
        body=serialize_doc(doc),
        headers={"type": "message.received", "x-attempt": 0}
    )
    socketio.emit("message:ack", serialize_doc(doc), room=session_id)
    return jsonify(serialize_doc(doc))

@app.route("/end_session", methods=["POST"])
def end_session():
    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    clear_redis_memory(session_id)
    return jsonify({"message": f"Session {session_id} cleared from Redis"})


@app.route("/")
def index():
    return "Flask-SocketIO is running!"


@socketio.on("connect")
def on_connect():
    emit("server:hello", {"ok": True, "msg": "Socket connected"})

@socketio.on("join")  # client tells us their session_id to get targeted updates
def on_join(data):
    session_id = (data or {}).get("session_id")
    if not session_id:
        emit("error", {"msg": "session_id required to join"})
        return
    join_room(session_id)
    emit("joined", {"session_id": session_id})

@socketio.on("leave")
def on_leave(data):
    session_id = (data or {}).get("session_id")
    if session_id:
        leave_room(session_id)
        emit("left", {"session_id": session_id})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5001, debug=True)
    socketio.run(app, port=5001, debug=True)
