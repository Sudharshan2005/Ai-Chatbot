import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")

genai.configure(api_key=GEMINI_API_KEY)

try:
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini model: {e}")

def get_gemini_response(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")