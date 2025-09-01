from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini import get_gemini_response
import uvicorn

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat_with_gemini(request: PromptRequest):
    try:
        response_text = get_gemini_response(request.prompt)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
