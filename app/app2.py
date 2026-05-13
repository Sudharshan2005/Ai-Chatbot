from flask import Flask, request, jsonify
import os, time, traceback
from datetime import datetime, timezone
from uuid import uuid4
import google.generativeai as genai
from db import (save_message, get_session_case_ids, get_messages_by_case_ids,
                serialize_doc, get_user_session_ids, get_messages_by_ids,
                sessions, agents, messages, get_message_by_id)
from redis_utils import load_redis_memory, save_redis_memory, clear_redis_memory
from mq import setup_topology, publish_event
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from supabase import create_client

load_dotenv()

# ── Gemini ────────────────────────────────────────────────────────────────────
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    bot = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    print("[OK] Gemini configured")
except Exception as e:
    print(f"[WARN] Gemini init failed: {e}")
    bot = None

# ── RabbitMQ (non-critical) ───────────────────────────────────────────────────
if os.getenv("AMQP_URL"):
    try:
        setup_topology()
        print("[OK] RabbitMQ topology ready")
    except Exception as e:
        print(f"[WARN] RabbitMQ unavailable: {e}")
else:
    print("[INFO] AMQP_URL not set — RabbitMQ disabled")

# ── Flask + SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "chatbot-secret")

_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in _origins_raw.split(",")]

CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}}, supports_credentials=True)
socketio = SocketIO(
    app,
    cors_allowed_origins=CORS_ORIGINS,
    cors_credentials=True,
    async_mode="eventlet",
    logger=False,
    engineio_logger=False,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_publish(body, headers=None):
    try:
        publish_event(body=body, headers=headers or {})
    except Exception as e:
        print(f"[WARN] RabbitMQ publish skipped: {e}")


def parse_timestamp(ts):
    if ts is None:
        return int(datetime.utcnow().timestamp() * 1000)
    if isinstance(ts, dict):
        ts_val = ts.get("$date", ts)
        if isinstance(ts_val, dict):
            return int(ts_val.get("$numberLong", 0))
        ts = ts_val
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, str):
        try:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.utcnow().timestamp() * 1000)


def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials missing")
    return create_client(url, key)


class CaseMemory:
    def __init__(self, initial_memory=None):
        self.memory = initial_memory or []

    def add_interaction(self, user_msg, bot_resp, intent):
        self.memory.append({
            "user": user_msg,
            "bot": bot_resp,
            "intent": intent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_context(self):
        return "\n".join(f"User: {m['user']}\nBot: {m['bot']}" for m in self.memory[-10:])


def analyze_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["hello", "hi", "hey"]):        return "greeting"
    if any(w in q for w in ["bye", "goodbye", "see you"]): return "farewell"
    if "order" in q or "status" in q:                      return "order_status"
    if "complaint" in q or "not working" in q:             return "complaint"
    if "help" in q or "how to" in q:                       return "faq_query"
    if "name" in q:                                         return "name_query"
    return "other"


def gemini_respond(user_message: str, context: str) -> str:
    if not bot:
        return "I'm having trouble connecting right now. Please try again shortly."
    prompt = (
        "You are a helpful and friendly customer support chatbot. "
        "Keep answers concise, warm, and empathetic.\n\n"
        f"Conversation so far:\n{context}\n\n"
        f"User: {user_message}\n"
        "Assistant:"
    )
    try:
        return bot.generate_content(prompt).text.strip()
    except Exception as e:
        print(f"[WARN] Gemini generation failed: {e}")
        return "I'm sorry, I couldn't process your request right now. Please try again."


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Chatbot API running"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        user_message = data.get("message", "").strip()
        session_id   = data.get("session_id") or str(uuid4())
        org_id       = data.get("org_id", "acme")
        channel      = data.get("channel", "web")
        request_id   = str(uuid4())
        case_id      = str(uuid4())
        message_id   = str(uuid4())
        t0           = time.perf_counter()

        # ── Check Supabase ticket ────────────────────────────────────────────
        ticket = None
        is_active_ticket = False
        is_assigned_to_agent = False
        try:
            sb = get_supabase()
            r = sb.table("tickets").select("*").eq("sessionId", session_id).execute()
            ticket = r.data[0] if r.data else None
            is_active_ticket     = bool(ticket and ticket.get("isActive"))
            is_assigned_to_agent = is_active_ticket and bool(ticket.get("escalatedTo"))
            print(f"[DEBUG] Ticket for {session_id}: {ticket}")
        except Exception as e:
            print(f"[WARN] Supabase ticket lookup failed: {e}")

        # ── Upsert session ───────────────────────────────────────────────────
        sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$push": {"message_ids": message_id},
             "$setOnInsert": {"user_id": user_id, "isActive": True}},
            upsert=True,
        )

        # ── Waiting for agent assignment ─────────────────────────────────────
        if is_active_ticket and not is_assigned_to_agent:
            waiting_doc = {
                "org_id": org_id, "user_id": user_id, "channel": channel,
                "session_id": session_id, "case_id": case_id, "message_id": message_id,
                "user_message": user_message,
                "response": "Your request has been escalated. An agent will be assigned shortly. Please wait.",
                "source": "system", "status": "waiting",
                "ticket": {"escalated": True, "ticket_id": None, "resolution_code": None},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_message(waiting_doc)
            socketio.emit("message:ack", serialize_doc(waiting_doc), room=session_id)
            return jsonify(serialize_doc(waiting_doc))

        # ── Forward to agent ─────────────────────────────────────────────────
        if is_assigned_to_agent:
            agent_id = ticket.get("escalatedTo")
            user_msg_doc = {
                "org_id": org_id, "user_id": user_id, "channel": channel,
                "session_id": session_id, "case_id": case_id, "message_id": message_id,
                "user_message": user_message, "response": None,
                "source": "user", "status": "open",
                "nlu": {"intent": "user_to_agent", "intent_confidence": 1.0,
                        "language": "en", "sentiment": "neutral", "tone": "neutral"},
                "ticket": {"escalated": True, "ticket_id": None, "resolution_code": None},
                "waiting_for_agent_response": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_message(user_msg_doc)
            socketio.emit("new_user_message", serialize_doc(user_msg_doc), room=f"agent_{agent_id}")
            safe_publish(serialize_doc(user_msg_doc), {"type": "user.message.to.agent", "x-attempt": 0})
            return jsonify(serialize_doc(user_msg_doc))

        # ── Load / build memory ──────────────────────────────────────────────
        memory_data = load_redis_memory(user_id, session_id)
        if memory_data and isinstance(memory_data, list):
            memory = CaseMemory(initial_memory=memory_data)
        else:
            case_ids = get_session_case_ids(session_id)
            past = get_messages_by_case_ids(case_ids)
            memory = CaseMemory(initial_memory=[
                {"user": m["user_message"], "bot": m.get("response", ""),
                 "intent": m.get("nlu", {}).get("intent", "other"),
                 "timestamp": m.get("timestamp", datetime.now(timezone.utc).isoformat())}
                for m in past if m.get("user_message")
            ])

        intent = analyze_intent(user_message)
        intent_confidence = 0.8 if intent != "other" else 0.5

        # ── Case memory hit ──────────────────────────────────────────────────
        for past in memory.memory:
            if past["user"].lower() == user_message.lower() and past.get("bot"):
                latency_ms = round((time.perf_counter() - t0) * 1000)
                doc = _build_doc(org_id, user_id, channel, session_id, case_id, message_id,
                                 request_id, user_message, past["bot"], "case_memory",
                                 intent, intent_confidence, 1.0, latency_ms)
                save_message(doc)
                save_redis_memory(user_id, session_id, memory)
                safe_publish(serialize_doc(doc), {"type": "message.received", "x-attempt": 0})
                socketio.emit("message:ack", serialize_doc(doc), room=session_id)
                return jsonify(serialize_doc(doc))

        # ── Gemini ───────────────────────────────────────────────────────────
        resp_text = gemini_respond(user_message, memory.get_context())
        memory.add_interaction(user_message, resp_text, intent)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        doc = _build_doc(org_id, user_id, channel, session_id, case_id, message_id,
                         request_id, user_message, resp_text, "gemini",
                         intent, intent_confidence, 0.4, latency_ms, model="gemini-2.5-flash")
        save_message(doc)
        save_redis_memory(user_id, session_id, memory)
        safe_publish(serialize_doc(doc), {"type": "message.received", "x-attempt": 0})
        socketio.emit("message:ack", serialize_doc(doc), room=session_id)
        return jsonify(serialize_doc(doc))

    except Exception as e:
        print("Error in /chat:", traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


def _build_doc(org_id, user_id, channel, session_id, case_id, message_id,
               request_id, user_message, response, source,
               intent, intent_confidence, similarity_score, latency_ms, model=None):
    return {
        "org_id": org_id, "user_id": user_id, "channel": channel,
        "session_id": session_id, "case_id": case_id, "message_id": message_id,
        "parent_message_id": None, "request_id": request_id, "direction": "inbound",
        "user_message": user_message, "response": response, "source": source,
        "status": "resolved",
        "nlu": {"intent": intent, "intent_confidence": intent_confidence,
                "language": "en", "sentiment": "neutral", "tone": "neutral"},
        "retrieval": {"kb_id": "default", "top_k_doc_ids": [],
                      "answer_confidence": similarity_score, "similarity_score": similarity_score},
        "llm": {"model": model, "latency_ms": latency_ms,
                "prompt_tokens": None, "completion_tokens": None},
        "ticket": {"escalated": False, "ticket_id": None, "resolution_code": None},
        "feedback": {"user_rating": None, "user_comment": None},
        "security": {"pii_redacted": True, "pii_types": []},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.route("/end_session", methods=["POST"])
def end_session():
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        user_id    = data.get("user_id")
        if not session_id or not user_id:
            return jsonify({"error": "session_id and user_id required"}), 400
        clear_redis_memory(session_id)
        sessions.update_one({"session_id": session_id, "user_id": user_id}, {"$set": {"isActive": False}})
        return jsonify({"message": f"Session {session_id} ended"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/user/sessions", methods=["GET"])
def get_user_sessions():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        session_ids = get_user_session_ids(user_id)
        historical_chats = {}
        for sid in session_ids:
            session = sessions.find_one({"session_id": sid, "user_id": user_id})
            if session and "message_ids" in session:
                msgs = get_messages_by_ids(session["message_ids"])
                historical_chats[sid] = [serialize_doc(m) for m in msgs]
            else:
                historical_chats[sid] = []
        return jsonify({"historical_chats": historical_chats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/agent/sessions", methods=["GET"])
def get_agent_sessions():
    try:
        agent_id = request.args.get("agent_id")
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400

        sb = get_supabase()
        tickets_resp = sb.table("tickets").select("*").eq("escalatedTo", agent_id).execute()
        tickets = tickets_resp.data or []

        sessions_data = []
        for ticket in tickets:
            session_id = ticket.get("sessionId")
            if not session_id:
                continue
            session = sessions.find_one({"session_id": session_id})
            if not session:
                continue

            msgs_list = list(messages.find({"session_id": session_id}).sort("timestamp", 1))
            processed = []
            for msg in msgs_list:
                if msg.get("user_message") is None and msg.get("source") == "agent":
                    processed.append({
                        "id": f"{msg.get('message_id')}-agent",
                        "role": "assistant", "content": msg.get("response", ""),
                        "createdAt": msg.get("timestamp"), "isAgent": True,
                    })
                else:
                    if msg.get("user_message"):
                        processed.append({
                            "id": msg.get("message_id"),
                            "role": "user", "content": msg.get("user_message"),
                            "createdAt": msg.get("timestamp"),
                        })
                    if msg.get("response"):
                        processed.append({
                            "id": f"{msg.get('message_id')}-response",
                            "role": "assistant", "content": msg.get("response"),
                            "createdAt": msg.get("timestamp"),
                            "isAgent": msg.get("source") == "agent",
                        })

            title = next(
                (m["content"][:50] + ("..." if len(m["content"]) > 50 else "")
                 for m in processed if m["role"] == "user"),
                "Untitled Session"
            )

            # Get agent name from MongoDB if available
            agent_doc = agents.find_one({"user_id": agent_id})
            sessions_data.append({
                "id": session_id, "title": title,
                "status": "active" if ticket.get("isActive") else "resolved",
                "priority": ticket.get("priority", "low"),
                "assignee": ticket.get("escalatedTo"),
                "assigneeName": agent_doc.get("name") if agent_doc else agent_id,
                "userName": ticket.get("userName"),
                "userEmail": ticket.get("userId") or session.get("user_id"),
                "tags": ticket.get("tags", []),
                "messages": processed,
                "createdAt": parse_timestamp(session.get("last_updated")),
                "updatedAt": parse_timestamp(ticket.get("updatedAt") or session.get("last_updated")),
                "closedAt": None if ticket.get("isActive") else parse_timestamp(
                    ticket.get("updatedAt") or session.get("last_updated")),
            })

        return jsonify({"sessions": sessions_data})
    except Exception as e:
        print("Error in /agent/sessions:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/agent/chat", methods=["POST"])
def agent_chat():
    try:
        data = request.json or {}
        agent_id   = data.get("agent_id")
        user_msg   = data.get("message", "")
        session_id = data.get("session_id")
        if not agent_id or not session_id:
            return jsonify({"error": "agent_id and session_id are required"}), 400

        session = sessions.find_one({"session_id": session_id})
        if not session:
            return jsonify({"error": "Session not found"}), 404

        msg_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        agent_doc = {
            "message_id": msg_id, "session_id": session_id,
            "user_message": None, "response": user_msg,
            "source": "agent", "role": "assistant", "agent_id": agent_id,
            "status": "resolved", "waiting_for_agent_response": False,
            "timestamp": now, "direction": "outbound",
        }
        messages.insert_one(agent_doc)
        sessions.update_one({"session_id": session_id}, {"$addToSet": {"message_ids": msg_id}})

        payload = serialize_doc(agent_doc)
        socketio.emit("agent_message", payload, room=session_id)
        socketio.emit("agent_message_sent", payload, room=f"agent_{agent_id}")
        safe_publish(payload, {"type": "agent.message.sent", "x-attempt": 0})
        return jsonify(payload)
    except Exception as e:
        print("Error in /agent/chat:", traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/agent/assign-session", methods=["POST"])
def assign_session():
    try:
        data = request.get_json() or {}
        agents.update_one(
            {"user_id": data.get("agent_id")},
            {"$addToSet": {"current_sessions": data.get("session_id")},
             "$set": {"updated_at": datetime.utcnow()}},
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/agent/remove-session", methods=["POST"])
def remove_session():
    try:
        data = request.get_json() or {}
        agents.update_one(
            {"user_id": data.get("agent_id")},
            {"$pull": {"current_sessions": data.get("session_id")},
             "$set": {"updated_at": datetime.utcnow()}},
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/agents", methods=["GET"])
def get_agents():
    try:
        result = [serialize_doc(a) for a in agents.find({})]
        return jsonify({"agents": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/sessions", methods=["GET"])
def get_all_sessions():
    try:
        result = [serialize_doc(s) for s in sessions.find()]
        return jsonify({"sessions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/message/<message_id>", methods=["GET"])
def get_message(message_id):
    try:
        msg = get_message_by_id(message_id)
        if msg:
            return jsonify(serialize_doc(msg))
        return jsonify({"error": "Message not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("server:hello", {"ok": True, "msg": "Socket connected"})

@socketio.on("disconnect")
def on_disconnect():
    pass

@socketio.on("join")
def on_join(data):
    sid = (data or {}).get("session_id")
    if not sid:
        emit("error", {"msg": "session_id required"}); return
    join_room(sid)
    emit("joined", {"session_id": sid})

@socketio.on("leave")
def on_leave(data):
    sid = (data or {}).get("session_id")
    if sid:
        leave_room(sid)
        emit("left", {"session_id": sid})

@socketio.on("agent_join")
def on_agent_join(data):
    aid = (data or {}).get("agent_id")
    if not aid:
        emit("error", {"msg": "agent_id required"}); return
    join_room(f"agent_{aid}")
    emit("agent_joined", {"agent_id": aid})

@socketio.on("agent_leave")
def on_agent_leave(data):
    aid = (data or {}).get("agent_id")
    if aid:
        leave_room(f"agent_{aid}")
        emit("agent_left", {"agent_id": aid})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
