# worker.py
import json, os, time, datetime, traceback
import pika
from tenacity import retry, stop_after_attempt, wait_exponential
from mq import get_connection, setup_topology, EXCHANGE_NAME, PROCESS_QUEUE
from db import save_message
from dotenv import load_dotenv
load_dotenv()

from flask_socketio import SocketIO

SOCKETIO_REDIS_URL = os.getenv("SOCKETIO_REDIS_URL")
socketio = SocketIO(message_queue=SOCKETIO_REDIS_URL)

PREFETCH = 10
MAX_ATTEMPTS = 5

def process_message(payload: dict) -> dict:
    result = dict(payload)
    result["worker_processed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    result["worker_node"] = os.getenv("HOSTNAME", "worker")


    # 🔔 Push to browsers connected to this session
    session_id = result.get("session_id")
    if session_id:
        socketio.emit("message:processed", result, room=session_id)

    return result

def _get_attempt(properties) -> int:
    headers = properties.headers or {}
    try:
        return int(headers.get("x-attempt", 0))
    except Exception:
        return 0

def _inc_headers(properties):
    headers = (properties.headers or {}).copy()
    headers["x-attempt"] = _get_attempt(properties) + 1
    return headers

def main():
    setup_topology()
    conn = get_connection()
    ch = conn.channel()
    ch.basic_qos(prefetch_count=PREFETCH)

    def on_message(ch, method, properties, body):
        print(f"📥 Received raw message on {method.routing_key}: {body}")
        try:
            payload = json.loads(body.decode("utf-8"))
            result = process_message(payload)
            print(f"✅ Processed message: {result}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print("Worker error:", e, traceback.format_exc())
            attempt = _get_attempt(properties)
            if attempt + 1 >= MAX_ATTEMPTS:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            else:
                headers = _inc_headers(properties)
                ch.basic_publish(
                    exchange=method.exchange or EXCHANGE_NAME,
                    routing_key=method.routing_key,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=2,
                        headers=headers
                    )
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=PROCESS_QUEUE, on_message_callback=on_message, auto_ack=False)
    print("Worker consuming…")
    ch.start_consuming()

if __name__ == "__main__":
    main()
