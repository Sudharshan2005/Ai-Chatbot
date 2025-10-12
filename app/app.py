from flask import Flask, request, jsonify
import joblib, os, time
from datetime import datetime, timezone
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from db import save_message, get_session_case_ids, get_messages_by_case_ids, serialize_doc, get_user_session_ids, get_messages_by_ids, sessions, agents, messages, get_message_by_id
from redis_utils import load_redis_memory, save_redis_memory, clear_redis_memory
from mq import setup_topology, publish_event
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import traceback
from sentence_transformers import SentenceTransformer, util
from supabase import create_client

load_dotenv()
df, X, embedding_model = joblib.load("../../chatbot.pkl")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
bot = genai.GenerativeModel(model_name='models/gemini-2.5-flash')
setup_topology()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "test-secret")
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"], cors_credentials=True, logger=True, engineio_logger=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

def parse_timestamp(ts):
    """Convert datetime, dict, or ISO string to milliseconds"""
    if ts is None:
        return int(datetime.utcnow().timestamp() * 1000)
    if isinstance(ts, dict):
        # MongoDB extended JSON format
        if '$date' in ts:
            ts_val = ts['$date']
            if isinstance(ts_val, dict) and '$numberLong' in ts_val:
                return int(ts_val['$numberLong'])
            else:
                ts = ts_val
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, str):
        return int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() * 1000)
    return int(datetime.utcnow().timestamp() * 1000)


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

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        supabase_client = create_client(supabase_url, supabase_key)
        
        ticket_response = supabase_client.table('tickets') \
            .select('*') \
            .eq('sessionId', session_id) \
            .execute()
        
        ticket = ticket_response.data[0] if ticket_response.data else None
        is_assigned_to_agent = ticket and ticket.get('escalatedTo') and ticket.get('isActive')

        # Save message_id to sessions collection
        sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            {
                "$push": {"message_ids": message_id},
                "$setOnInsert": {
                    "user_id": user_id,
                    "isActive": True
                }
            },
            upsert=True
        )

        if is_assigned_to_agent:
            agent_id = ticket.get('escalatedTo')
            
            # Save user message
            user_message_doc = {
                "org_id": org_id, "user_id": user_id, "channel": channel,
                "session_id": session_id, "case_id": case_id, "message_id": message_id,
                "parent_message_id": None, "request_id": request_id, "direction": "inbound",
                "user_message": user_message, "response": None, "source": "user",
                "status": "open",
                "nlu": {"intent": "user_to_agent", "intent_confidence": 1.0, "language": "en", "sentiment": "neutral", "tone": "neutral"},
                "retrieval": {"kb_id": "default", "top_k_doc_ids": [], "answer_confidence": 1.0, "similarity_score": 1.0},
                "llm": {"model": None, "latency_ms": 0, "prompt_tokens": None, "completion_tokens": None},
                "ticket": {"escalated": True, "ticket_id": None, "resolution_code": None},
                "feedback": {"user_rating": None, "user_comment": None},
                "security": {"pii_redacted": True, "pii_types": []},
                "waiting_for_agent_response": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            save_message(user_message_doc)
            
            # Emit to Socket.IO for real-time updates
            socketio.emit("user_message", serialize_doc(user_message_doc), room=session_id)
            # Notify the assigned agent
            socketio.emit("new_user_message", serialize_doc(user_message_doc), room=f"agent_{agent_id}")
            
            publish_event(
                body=serialize_doc(user_message_doc),
                headers={"type": "user.message.to.agent", "x-attempt": 0}
            )
            
            return jsonify({
                "message": "Message forwarded to agent",
                "assigned_agent": agent_id,
                "user_message": user_message,
                "session_id": session_id
            })

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
        try:
            query_emb = model.encode([user_message], convert_to_tensor=True)
            answer_emb = model.encode([resp_text], convert_to_tensor=True)
            sim = float(util.cos_sim(query_emb, answer_emb).item())
        except Exception as e:
            print("Similarity computation failed:", e)
            sim = 0.3
        if sim >= 0.5:
            similarity_score = sim
        else:
            similarity_score = sim * 0.8
        latency_ms = round((time.perf_counter() - t0) * 1000)
        doc = {
            "org_id": org_id, "user_id": user_id, "channel": channel,
            "session_id": session_id, "case_id": case_id, "message_id": message_id,
            "parent_message_id": None, "request_id": request_id, "direction": "inbound",
            "user_message": user_message, "response": resp_text, "source": "gemini", "status": "open",
            "nlu": {"intent": intent, "intent_confidence": intent_confidence, "language": "en", "sentiment": "neutral", "tone": "neutral"},
            "retrieval": {"kb_id": "default", "top_k_doc_ids": [], "answer_confidence": similarity_score, "similarity_score": similarity_score},
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
        sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$set": {"isActive": False}}
        )
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
    
@app.route('/admin/sessions', methods=['GET'])
def get_all_sessions():
    try:
        # Fetch all session documents from MongoDB
        all_sessions = list(sessions.find())

        # Serialize each document (convert ObjectId to str)
        serialized_sessions = [serialize_doc(session) for session in all_sessions]

        return jsonify({'sessions': serialized_sessions}), 200

    except Exception as e:
        print("Error in /admin/sessions:", str(e))
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/admin/message/<message_id>', methods=['GET'])
def get_message(message_id):
    try:
        message = get_message_by_id(message_id)
        if message:
            message['_id'] = str(message['_id'])
            return jsonify(message)
        else:
            return jsonify({'error': 'Message not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/agents', methods=['GET'])
def get_agents():
    try:
        agents_list = list(agents.find({}))
        for agent in agents_list:
            agent['_id'] = str(agent['_id'])
        return jsonify({'agents': agents_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agent/assign-session', methods=['POST'])
def assign_session_to_agent():
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        session_id = data.get('session_id')
        
        result = agents.update_one(
            {'user_id': agent_id},
            {
                '$addToSet': {'current_sessions': session_id},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        
        if result.modified_count:
            return jsonify({'success': True, 'message': 'Session assigned'})
        else:
            return jsonify({'error': 'Agent not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agent/remove-session', methods=['POST'])
def remove_session_from_agent():
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        session_id = data.get('session_id')
        
        # Remove session from agent's current_sessions
        result = agents.update_one(
            {'user_id': agent_id},
            {
                '$pull': {'current_sessions': session_id},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        
        return jsonify({'success': True, 'message': 'Session removed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agent/sessions', methods=['GET'])
def get_agent_sessions():
    try:
        agent_id = request.args.get('agent_id')
        
        if not agent_id:
            return jsonify({'error': 'agent_id is required'}), 400
        
        # Get agent to find their sessions
        agent = agents.find_one({'user_id': agent_id})
        if not agent:
            return jsonify({'message': 'No Agent Found' ,'sessions': []})
        
        # Get tickets assigned to this agent from Supabase
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        supabase_client = create_client(supabase_url, supabase_key)
        
        # Get all tickets assigned to this agent
        tickets_response = supabase_client.table('tickets') \
            .select('*') \
            .eq('escalatedTo', agent_id) \
            .execute()
        
        tickets = tickets_response.data if tickets_response.data else []
        
        sessions_data = []
        
        for ticket in tickets:
            session_id = ticket.get('sessionId')
            if not session_id:
                continue
                
            # Get session from MongoDB
            session = sessions.find_one({'session_id': session_id})
            if not session:
                continue
            
            # Get messages for this session
            messages_list = list(messages.find({'session_id': session_id}))
            
            # Process messages
            processed_messages = []
            for msg in messages_list:
                # Add user message
                if msg.get('user_message'):
                    processed_messages.append({
                        'id': msg.get('message_id'),
                        'role': 'user',
                        'content': msg.get('user_message'),
                        'createdAt': msg.get('timestamp')
                    })
                
                # Add assistant response
                if msg.get('response'):
                    processed_messages.append({
                        'id': f"{msg.get('message_id')}-response",
                        'role': 'assistant',
                        'content': msg.get('response'),
                        'createdAt': msg.get('timestamp')
                    })
            
            # Sort messages by timestamp
            processed_messages.sort(key=lambda x: x['createdAt'])
            
            # Determine title from first user message
            title = "Untitled Session"
            for msg in processed_messages:
                if msg['role'] == 'user':
                    user_message = msg['content']
                    title = user_message[:50] + "..." if len(user_message) > 50 else user_message
                    break
            
            # Get user info
            user_email = ticket.get('userId') or session.get('user_id')
            user_name = ticket.get('userName')
            
            # Create session object
            session_data = {
                'id': session_id,
                'title': title,
                'status': 'active' if ticket.get('isActive') else 'resolved',
                'priority': ticket.get('priority', 'low'),
                'assignee': ticket.get('escalatedTo'),
                'assigneeName': agent.get('name'),
                'userName': user_name,
                'userEmail': user_email,
                'tags': ticket.get('tags', []),
                'messages': processed_messages,
                'createdAt': parse_timestamp(session.get('last_updated')),
                'updatedAt': parse_timestamp(ticket.get('updatedAt') or session.get('last_updated')),
                'closedAt': None if ticket.get('isActive') else parse_timestamp(ticket.get('updatedAt') or session.get('last_updated'))
            }
            
            # Convert timestamps to milliseconds
            try:
                if session_data['createdAt']:
                    if isinstance(session_data['createdAt'], dict) and '$numberLong' in session_data['createdAt']:
                        session_data['createdAt'] = int(session_data['createdAt']['$numberLong'])
                    else:
                        session_data['createdAt'] = int(datetime.fromisoformat(session_data['createdAt'].replace('Z', '+00:00')).timestamp() * 1000)
                
                if session_data['updatedAt']:
                    if isinstance(session_data['updatedAt'], dict) and '$numberLong' in session_data['updatedAt']:
                        session_data['updatedAt'] = int(session_data['updatedAt']['$numberLong'])
                    else:
                        session_data['updatedAt'] = int(datetime.fromisoformat(session_data['updatedAt'].replace('Z', '+00:00')).timestamp() * 1000)
                
                if session_data['closedAt']:
                    if isinstance(session_data['closedAt'], dict) and '$numberLong' in session_data['closedAt']:
                        session_data['closedAt'] = int(session_data['closedAt']['$numberLong'])
                    else:
                        session_data['closedAt'] = int(datetime.fromisoformat(session_data['closedAt'].replace('Z', '+00:00')).timestamp() * 1000)
            except (ValueError, AttributeError) as e:
                print(f"Error parsing timestamps for session {session_id}: {e}")
                # Set fallback timestamps
                current_time = int(datetime.utcnow().timestamp() * 1000)
                session_data['createdAt'] = session_data.get('createdAt') or current_time
                session_data['updatedAt'] = session_data.get('updatedAt') or current_time
            
            sessions_data.append(session_data)
        
        return jsonify({'sessions': sessions_data})
        
    except Exception as e:
        print(f"Error in get_agent_sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/agent/chat", methods=["POST"])
def agent_chat():
    try:
        data = request.json or {}
        agent_id = data.get("agent_id")
        user_message = data.get("message", "")
        session_id = data.get("session_id")
        
        if not agent_id or not session_id:
            return jsonify({"error": "agent_id and session_id are required"}), 400
        
        # Verify agent has access to this session
        agent = agents.find_one({'user_id': agent_id})
        if not agent or session_id not in agent.get('current_sessions', []):
            return jsonify({"error": "Agent not assigned to this session"}), 403
        
        # Verify session exists and is active
        session = sessions.find_one({'session_id': session_id})
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Get the latest user message that's waiting for agent response
        latest_user_message = messages.find_one({
            'session_id': session_id,
            'waiting_for_agent_response': True
        }, sort=[('timestamp', -1)])  # Get the most recent one
        
        if not latest_user_message:
            return jsonify({"error": "No user message found waiting for response"}), 404
        
        # Update the existing document with agent's response
        update_data = {
            "response": user_message,
            "source": "agent",
            "status": "resolved",
            "waiting_for_agent_response": False,  # Remove the flag
            "agent_id": agent_id,
            "llm.model": "agent",
            "timestamp": datetime.now(timezone.utc).isoformat()  # Update timestamp
        }
        
        # Perform the update
        result = messages.update_one(
            {'message_id': latest_user_message['message_id']},
            {'$set': update_data}
        )
        
        if result.modified_count == 0:
            return jsonify({"error": "Failed to update message with agent response"}), 500
        
        # Fetch the updated document
        updated_message = messages.find_one({'message_id': latest_user_message['message_id']})
        
        # Update session with the message_id (if not already there)
        sessions.update_one(
            {"session_id": session_id},
            {"$addToSet": {"message_ids": latest_user_message['message_id']}}
        )
        
        # Emit to Socket.IO so frontend updates in real-time
        socketio.emit("agent_message", serialize_doc(updated_message), room=session_id)
        
        # Also emit to agent's room for their dashboard
        socketio.emit("agent_message_sent", serialize_doc(updated_message), room=f"agent_{agent_id}")
        
        publish_event(
            body=serialize_doc(updated_message),
            headers={"type": "agent.message.sent", "x-attempt": 0}
        )
        
        return jsonify(serialize_doc(updated_message))
        
    except Exception as e:
        print("Error in /agent/chat:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/session/<session_id>/status", methods=["GET"])
def get_session_status(session_id):
    try:
        # Check if session is assigned to an agent
        session = sessions.find_one({'session_id': session_id})
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Check Supabase for ticket assignment
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        supabase_client = create_client(supabase_url, supabase_key)
        
        ticket_response = supabase_client.table('tickets') \
            .select('*') \
            .eq('sessionId', session_id) \
            .execute()
        
        ticket = ticket_response.data[0] if ticket_response.data else None
        
        is_assigned = False
        assigned_agent = None
        agent_name = None
        
        if ticket and ticket.get('escalatedTo'):
            is_assigned = True
            assigned_agent = ticket.get('escalatedTo')
            # Get agent name
            agent = agents.find_one({'user_id': assigned_agent})
            agent_name = agent.get('name') if agent else assigned_agent
        
        return jsonify({
            "session_id": session_id,
            "is_assigned_to_agent": is_assigned,
            "assigned_agent": assigned_agent,
            "assigned_agent_name": agent_name,
            "is_active": ticket.get('isActive') if ticket else True
        })
        
    except Exception as e:
        print("Error in /session/status:", str(e))
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

@socketio.on("agent_join")
def on_agent_join(data):
    agent_id = (data or {}).get("agent_id")
    if not agent_id:
        print("SocketIO agent_join failed: missing agent_id")
        emit("error", {"msg": "agent_id required to join agent room"})
        return
    join_room(f"agent_{agent_id}")
    print(f"Agent {agent_id} joined their room")
    emit("agent_joined", {"agent_id": agent_id})

@socketio.on("agent_leave")
def on_agent_leave(data):
    agent_id = (data or {}).get("agent_id")
    if agent_id:
        leave_room(f"agent_{agent_id}")
        print(f"Agent {agent_id} left their room")
        emit("agent_left", {"agent_id": agent_id})

if __name__ == "__main__":
    socketio.run(app, port=5001, debug=True)