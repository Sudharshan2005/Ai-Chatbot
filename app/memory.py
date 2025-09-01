import datetime

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