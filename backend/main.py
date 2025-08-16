from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    enhanced_prompt: str

@app.post("/enhance", response_model=PromptResponse)
async def enhance_prompt(request: PromptRequest):
    try:
        enhancement_instruction = (
            "You are a prompt enhancer. Improve the following prompt to be more "
            "clear, specific, and effective for generating high-quality AI responses. "
            "Return only the enhanced prompt."
        )
        full_prompt = f"{enhancement_instruction}\n\nOriginal Prompt: {request.prompt}"

        response = model.generate_content(full_prompt)
        enhanced = response.text.strip()

        return PromptResponse(enhanced_prompt=enhanced)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class ChatMessage(BaseModel):
    role: str        # "user" | "assistant"
    text: str

class ChatRequest(BaseModel):
    history: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Turn the conversation history into a single prompt for Gemini
    and return the assistant's reply.
    """
    try:
        # Build a simple prompt from the history
        prompt_lines = []
        for msg in req.history:
            speaker = "User" if msg.role == "user" else "Assistant"
            prompt_lines.append(f"{speaker}: {msg.text}")
        prompt_lines.append("Assistant:")
        prompt = "\n".join(prompt_lines)

        resp = model.generate_content(prompt)
        return ChatResponse(reply=resp.text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)