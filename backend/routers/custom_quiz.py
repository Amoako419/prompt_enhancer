from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import json
import os
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to allow importing services
sys.path.append(str(Path(__file__).parent.parent))
from services.gemini_service import generate_quiz, QuizGenerationRequest

# Custom quiz model
class QuizOption(BaseModel):
    text: str

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: int
    explanation: Optional[str] = None
    type: str = "multiple_choice"

class CustomQuiz(BaseModel):
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    difficulty: str
    duration: int
    questions: List[QuizQuestion]
    created_at: Optional[str] = None
    user_id: Optional[str] = None

router = APIRouter(
    prefix="/custom-quiz",
    tags=["custom-quiz"],
    responses={404: {"description": "Not found"}},
)

# In-memory storage (replace with database in production)
CUSTOM_QUIZZES = []

# File storage path (for persistence between restarts)
QUIZZES_FILE = "custom_quizzes.json"

# Load quizzes from file on startup
def load_quizzes_from_file():
    try:
        if os.path.exists(QUIZZES_FILE):
            with open(QUIZZES_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading quizzes: {e}")
    return []

# Save quizzes to file
def save_quizzes_to_file():
    try:
        with open(QUIZZES_FILE, 'w') as f:
            json.dump(CUSTOM_QUIZZES, f)
    except Exception as e:
        print(f"Error saving quizzes: {e}")

# Initialize quizzes
CUSTOM_QUIZZES = load_quizzes_from_file()

@router.post("/create", response_model=CustomQuiz)
async def create_custom_quiz(quiz: CustomQuiz):
    """
    Create a new custom quiz
    """
    # Add metadata
    quiz_id = f"quiz_{len(CUSTOM_QUIZZES) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    quiz.id = quiz_id
    quiz.created_at = datetime.now().isoformat()
    
    # Store quiz
    CUSTOM_QUIZZES.append(quiz.dict())
    save_quizzes_to_file()
    
    return quiz

@router.get("/list", response_model=List[CustomQuiz])
async def list_custom_quizzes():
    """
    List all custom quizzes
    """
    return CUSTOM_QUIZZES

@router.get("/{quiz_id}", response_model=CustomQuiz)
async def get_custom_quiz(quiz_id: str):
    """
    Get a specific custom quiz by ID
    """
    for quiz in CUSTOM_QUIZZES:
        if quiz.get("id") == quiz_id:
            return quiz
    raise HTTPException(status_code=404, detail="Quiz not found")

@router.delete("/{quiz_id}")
async def delete_custom_quiz(quiz_id: str):
    """
    Delete a custom quiz by ID
    """
    global CUSTOM_QUIZZES
    
    initial_length = len(CUSTOM_QUIZZES)
    CUSTOM_QUIZZES = [quiz for quiz in CUSTOM_QUIZZES if quiz.get("id") != quiz_id]
    
    if len(CUSTOM_QUIZZES) == initial_length:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    save_quizzes_to_file()
    return {"message": "Quiz deleted successfully"}

@router.post("/generate", response_model=dict)
async def generate_ai_quiz(request: QuizGenerationRequest):
    """
    Generate a quiz with AI using specified parameters
    """
    try:
        # Call Gemini service to generate the quiz
        generated_quiz = generate_quiz(request)
        
        # Format the response to match the frontend's expected format
        return {
            "title": generated_quiz.title,
            "questions": [
                {
                    "question": q.question,
                    "options": q.options,
                    "correct_index": q.correct_index,
                    "explanation": q.explanation
                } for q in generated_quiz.questions
            ]
        }
    except Exception as e:
        # Log error for debugging
        print(f"Quiz generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")
