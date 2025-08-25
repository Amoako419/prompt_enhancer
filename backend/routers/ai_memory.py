from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

import google.generativeai as genai
import json
import os

from database import get_db
from models.models import Memory, User
from auth.dependencies import get_current_active_user

router = APIRouter(
    prefix="/api/ai",
    tags=["ai"],
    responses={401: {"description": "Unauthorized"}},
)

# Pydantic models for request/response
class PromptRequest(BaseModel):
    prompt: str
    save_history: bool = True
    use_history: bool = True
    history_limit: Optional[int] = 5

class PromptResponse(BaseModel):
    response: str
    
# Configure Gemini (uses the config from main.py)
model = genai.GenerativeModel("gemini-2.0-flash")


async def get_user_history(db: AsyncSession, user_id: int, limit: int = 5):
    """Get user's conversation history from memory"""
    result = await db.execute(
        select(Memory)
        .filter(Memory.user_id == user_id, Memory.key.startswith("chat_history_"))
        .order_by(Memory.updated_at.desc())
        .limit(limit)
    )
    
    history_entries = result.scalars().all()
    
    # Convert to list of messages format
    history = []
    for entry in history_entries:
        try:
            history_item = json.loads(entry.value)
            history.append(history_item)
        except Exception:
            continue
            
    # Reverse to get chronological order
    history.reverse()
    
    return history


@router.post("/generate", response_model=PromptResponse)
async def generate_enhanced_prompt(
    request: PromptRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate an enhanced response using Gemini, with optional memory integration"""
    
    # Initialize conversation
    chat = model.start_chat(history=[])
    
    # Add user's conversation history if requested
    if request.use_history:
        history = await get_user_history(db, current_user.id, request.history_limit)
        
        # Add history to chat
        for item in history:
            if item.get("role") == "user":
                chat.send_message(item.get("content", ""))
            elif item.get("role") == "model":
                # We can't directly add model responses to history, 
                # but they'll be included in the next send_message call
                pass
    
    # Send the user prompt
    response = chat.send_message(request.prompt)
    response_text = response.text
    
    # Save to history if requested
    if request.save_history:
        # Create a timestamp-based key
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        
        # Save user message
        user_history = Memory(
            user_id=current_user.id,
            key=f"chat_history_{timestamp}_user",
            value=json.dumps({"role": "user", "content": request.prompt})
        )
        
        # Save model response
        model_history = Memory(
            user_id=current_user.id,
            key=f"chat_history_{timestamp}_model",
            value=json.dumps({"role": "model", "content": response_text})
        )
        
        db.add(user_history)
        db.add(model_history)
        await db.commit()
    
    return {"response": response_text}
