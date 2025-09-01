# mq.py
import json, os
import pika
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()
AMQP_URL = os.getenv("AMQP_URL")

EXCHANGE_NAME = "chat.events"
EXCHANGE_TYPE = "topic"
PROCESS_QUEUE = "chat.process"
DLQ_QUEUE = "chat.process.dlq"
ROUTING_INBOUND = "message.received"

def _params():
    return pika.URLParameters(AMQP_URL)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_connection():
    return pika.BlockingConnection(_params())

def setup_topology():
    conn = get_connection()
    ch = conn.channel()

    ch.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE, durable=True)

    ch.queue_declare(queue=DLQ_QUEUE, durable=True)

    ch.queue_declare(
        queue=PROCESS_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": EXCHANGE_NAME,
            "x-dead-letter-routing-key": ROUTING_INBOUND
        }
    )
    ch.queue_bind(queue=PROCESS_QUEUE, exchange=EXCHANGE_NAME, routing_key=ROUTING_INBOUND)

    conn.close()

def publish_event(body: dict, routing_key=ROUTING_INBOUND, headers: dict | None = None):
    conn = get_connection()
    ch = conn.channel()
    ch.confirm_delivery()

    props = pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,
        headers=headers or {}
    )
    ch.basic_publish(
        exchange=EXCHANGE_NAME,
        routing_key=routing_key,
        body=json.dumps(body),
        properties=props,
        mandatory=True
    )
    conn.close()
