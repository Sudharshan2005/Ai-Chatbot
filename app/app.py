from flask import Flask, request, jsonify
import joblib, os, time
from datetime import datetime, timezone
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from db import save_message, get_session_case_ids, get_messages_by_case_ids, serialize_doc, get_user_session_ids, get_messages_by_ids, sessions
from redis_utils import load_redis_memory, save_redis_memory, clear_redis_memory
from mq import setup_topology, publish_event
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import traceback

load_dotenv()
df, X, embedding_model = joblib.load("../../chatbot.pkl")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
bot = genai.GenerativeModel(model_name='models/gemini-2.5-flash')
setup_topology()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "test-secret")
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"], cors_credentials=True, logger=True, engineio_logger=True)

class CaseMemory:
    def __init__(self, initial_memory=None):
        self.memory = initial_memory or []

    def add_interaction(self, user_message, bot_response, intent):
        self.memory.append({
            "user": user_message,
            "bot": bot_response,
            "intent": intent,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def get_context(self):
        return "\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in self.memory])

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

@app.errorhandler(Exception)
def handle_error(error):
    print("Server error:", str(error))
    print(traceback.format_exc())
    return jsonify({"error": "Internal server error", "details": str(error)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        user_message = data.get("message", "")
        session_id = data.get("session_id") or str(uuid4())
        org_id = data.get("org_id", "acme")
        channel = data.get("channel", "web")
        request_id = data.get("request_id") or str(uuid4())
        case_id = str(uuid4())
        message_id = str(uuid4())
        t0 = time.perf_counter()

        print(f"Received message for session_id: {session_id}, user_id: {user_id}, message: {user_message}")

        # Save message_id to sessions collection
        sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$push": {"message_ids": message_id}, "$setOnInsert": {"user_id": user_id}},
            upsert=True
        )

        # Load memory from Redis
        memory_data = load_redis_memory(user_id, session_id)
        memory = CaseMemory(initial_memory=memory_data if isinstance(memory_data, list) else [])
        if not memory_data:
            case_ids = get_session_case_ids(session_id)
            past_messages = get_messages_by_case_ids(case_ids)
            initial_memory = [
                {
                    "user": m["user_message"],
                    "bot": m.get("response", ""),
                    "intent": m.get("nlu", {}).get("intent", "other"),
                    "timestamp": m.get("timestamp", datetime.now(timezone.utc).isoformat())
                }
                for m in past_messages
            ]
            memory = CaseMemory(initial_memory=initial_memory)

        intent = analyze_intent(user_message)
        intent_confidence = 0.7 if intent != "other" else 0.5

        # Check memory for matching user message
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                save_message(doc)
                save_redis_memory(user_id, session_id, memory)
                publish_event(
                    body=serialize_doc(doc),
                    headers={"type": "message.received", "x-attempt": 0}
                )
                socketio.emit("message:ack", serialize_doc(doc), room=session_id)
                return jsonify(serialize_doc(doc))

        # Try dataset answer
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
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            save_message(doc)
            save_redis_memory(user_id, session_id, memory)
            publish_event(
                body=serialize_doc(doc),
                headers={"type": "message.received", "x-attempt": 0}
            )
            socketio.emit("message:ack", serialize_doc(doc), room=session_id)
            return jsonify(serialize_doc(doc))

        # Generate response with Gemini
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        save_message(doc)
        save_redis_memory(user_id, session_id, memory)
        publish_event(
            body=serialize_doc(doc),
            headers={"type": "message.received", "x-attempt": 0}
        )
        socketio.emit("message:ack", serialize_doc(doc), room=session_id)
        return jsonify(serialize_doc(doc))
    except Exception as e:
        print("Error in /chat:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/end_session", methods=["POST"])
def end_session():
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        if not session_id or not user_id:
            return jsonify({"error": "session_id and user_id required"}), 400

        clear_redis_memory(session_id)
        return jsonify({"message": f"Session {session_id} cleared from Redis"})
    except Exception as e:
        print("Error in /end_session:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/user/sessions", methods=["GET"])
def get_user_sessions():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        session_ids = get_user_session_ids(user_id)
        historical_chats = {}
        for session_id in session_ids:
            session = sessions.find_one({"session_id": session_id, "user_id": user_id})
            if session and "message_ids" in session:
                message_ids = session["message_ids"]
                messages = get_messages_by_ids(message_ids)
                historical_chats[session_id] = [serialize_doc(m) for m in messages]
            else:
                historical_chats[session_id] = []

        return jsonify({"historical_chats": historical_chats})
    except Exception as e:
        print("Error in /user/sessions:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/")
def index():
    return "Flask-SocketIO is running!"

@socketio.on("connect")
def on_connect():
    print("SocketIO client connected")
    emit("server:hello", {"ok": True, "msg": "Socket connected"})

@socketio.on("disconnect")
def on_disconnect():
    print("SocketIO client disconnected")

@socketio.on("join")
def on_join(data):
    session_id = (data or {}).get("session_id")
    if not session_id:
        print("SocketIO join failed: missing session_id")
        emit("error", {"msg": "session_id required to join"})
        return
    join_room(session_id)
    print(f"SocketIO client joined room: {session_id}")
    emit("joined", {"session_id": session_id})

@socketio.on("leave")
def on_leave(data):
    session_id = (data or {}).get("session_id")
    if session_id:
        leave_room(session_id)
        print(f"SocketIO client left room: {session_id}")
        emit("left", {"session_id": session_id})

if __name__ == "__main__":
    socketio.run(app, port=5001, debug=True)