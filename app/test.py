from flask import Flask
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.config["SECRET_KEY"] = "test-secret"
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for testing

@app.route("/")
def home():
    return "WebSocket Test Server Running"

@socketio.on("connect")
def test_connect():
    print("✅ Client connected")
    send("Hello from Flask server!", broadcast=True)

@socketio.on("message")
def handle_message(msg):
    print(f"📩 Received message: {msg}")
    send(f"Echo: {msg}")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
