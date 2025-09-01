"""Basic connection example.
"""

import redis

r = redis.Redis(
    host='redis-10914.c305.ap-south-1-1.ec2.redns.redis-cloud.com',
    port=10914,
    decode_responses=False,
    username="default",
    password="VJVvAaqi3v4ydNvyUcKkbWzrtUACryx2",
)

print(r.get("session:786a43e0-4ca8-4170-817d-0cc3ab43b45a"))