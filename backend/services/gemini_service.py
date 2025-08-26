from typing import Dict, List, Optional
import os
import json
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()
# Get API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini AI
genai.configure(api_key=API_KEY)

class QuizGenerationRequest(BaseModel):
    category: str
    difficulty: str
    numQuestions: int
    topic: Optional[str] = None
    title: str
    
class GeneratedQuestion(BaseModel):
    question: str
    options: List[str]
    correct_index: int
    explanation: Optional[str] = None

class GeneratedQuiz(BaseModel):
    title: str
    questions: List[GeneratedQuestion]

def generate_quiz(request: QuizGenerationRequest) -> GeneratedQuiz:
    """
    Generate a quiz using Google's Gemini AI
    """
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    # Prepare the prompt
    specific_topic = f" specifically about {request.topic}" if request.topic else ""
    
    prompt = f"""
    Create a {request.difficulty} level quiz about {request.category}{specific_topic} with {request.numQuestions} multiple-choice questions.
    
    For each question:
    1. Create a clear, concise question
    2. Provide exactly 4 options
    3. Indicate which option is correct (0-indexed)
    4. Provide a brief explanation for the correct answer
    
    Return the quiz in this exact JSON format (no explanation outside the JSON):
    {{
      "title": "{request.title}",
      "questions": [
        {{
          "question": "Question text here",
          "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
          "correct_index": 0,
          "explanation": "Explanation for why this answer is correct"
        }}
      ]
    }}
    
    Make sure all answers are accurate and appropriate for {request.difficulty} level.
    """
    
    try:
        # Generate quiz content
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text
        
        # Find the start and end of JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            quiz_data = json.loads(json_str)
            
            # Parse and validate the response
            return GeneratedQuiz(**quiz_data)
        else:
            raise ValueError("Could not extract valid JSON from the AI response")
            
    except Exception as e:
        print(f"Error generating quiz: {e}")
        raise Exception(f"Failed to generate quiz: {str(e)}")
