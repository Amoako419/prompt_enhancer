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

class SqlRequest(BaseModel):
    english_query: str

class SqlResponse(BaseModel):
    sql_query: str

class DataExplorationRequest(BaseModel):
    description: str
    analysis_type: str  # "eda", "statistical", "anomaly", "timeseries", "visualization"

class DataExplorationResponse(BaseModel):
    code: str

class SkillAssessmentRequest(BaseModel):
    category: str      # "python", "sql", "statistics", "machine_learning", "data_visualization", "data_engineering"
    difficulty: str    # "beginner", "intermediate", "advanced"
    num_questions: int # number of questions to generate

class Question(BaseModel):
    question: str
    options: List[str]
    correct_answer: int  # index of correct option (0-based)
    explanation: str
    type: str           # "multiple_choice", "code_analysis", "scenario"

class SkillAssessmentResponse(BaseModel):
    questions: List[Question]
    category: str
    difficulty: str
    duration_minutes: int

class EvaluationRequest(BaseModel):
    category: str
    difficulty: str
    answers: List[int]  # user's selected answers (indices)
    questions: List[Question]  # original questions for scoring

class EvaluationResponse(BaseModel):
    score: int          # percentage score
    correct_count: int
    total_questions: int
    performance_level: str  # "beginner", "intermediate", "advanced", "expert"
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    detailed_feedback: str

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

@app.post("/to-sql", response_model=SqlResponse)
async def convert_to_sql(request: SqlRequest):
    """
    Convert plain English query to SQL statement
    """
    try:
        sql_instruction = (
            "You are an expert SQL generator. Convert the following plain English query "
            "into a well-formatted SQL statement. Only return the SQL query without any "
            "explanations or additional text. Assume common table names like 'users', "
            "'orders', 'products', 'customers' etc. unless specified otherwise."
        )
        full_prompt = f"{sql_instruction}\n\nEnglish Query: {request.english_query}"

        response = model.generate_content(full_prompt)
        sql_query = response.text.strip()
        
        # Clean up the response to remove any markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return SqlResponse(sql_query=sql_query.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-exploration", response_model=DataExplorationResponse)
async def generate_data_exploration(request: DataExplorationRequest):
    """
    Generate data exploration code based on dataset description and analysis type
    """
    try:
        analysis_prompts = {
            "eda": """You are an expert data scientist. Generate comprehensive Python code for Exploratory Data Analysis (EDA). 
Include: data overview (shape, info, describe), null values analysis, data types, unique values, 
distribution plots (histograms, box plots), correlation analysis, and basic insights.
Use pandas, matplotlib, seaborn, and numpy. Make the code well-commented and production-ready.""",
            
            "statistical": """You are an expert statistician. Generate Python code for statistical analysis.
Include: descriptive statistics, normality tests, hypothesis testing, confidence intervals, 
t-tests, chi-square tests, ANOVA where appropriate. Use scipy.stats, pandas, and numpy.
Make the code well-commented and include interpretation of results.""",
            
            "anomaly": """You are an expert in anomaly detection. Generate Python code for outlier detection.
Include: statistical methods (IQR, Z-score), isolation forest, local outlier factor, 
one-class SVM, and visualization of anomalies. Use sklearn, pandas, matplotlib, and seaborn.
Make the code modular and well-commented.""",
            
            "timeseries": """You are an expert in time series analysis. Generate Python code for time series analysis.
Include: trend analysis, seasonality decomposition, stationarity tests, autocorrelation, 
basic forecasting with moving averages or exponential smoothing. Use pandas, matplotlib, 
statsmodels, and seaborn. Make the code comprehensive and well-documented.""",
            
            "visualization": """You are an expert in data visualization. Generate Python code for comprehensive data visualization.
Include: distribution plots, scatter plots, correlation heatmaps, box plots, violin plots,
pair plots, and dashboard-style layouts. Use matplotlib, seaborn, and plotly.
Make visualizations publication-ready with proper titles, labels, and styling."""
        }
        
        instruction = analysis_prompts.get(request.analysis_type, analysis_prompts["eda"])
        full_prompt = f"{instruction}\n\nDataset Description: {request.description}\n\nGenerate Python code:"

        response = model.generate_content(full_prompt)
        code = response.text.strip()
        
        # Clean up the response to remove any markdown formatting
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return DataExplorationResponse(code=code.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skill-assessment/generate", response_model=SkillAssessmentResponse)
async def generate_skill_assessment(request: SkillAssessmentRequest):
    """
    Generate skill assessment questions for a specific category and difficulty
    """
    try:
        category_prompts = {
            "python": {
                "beginner": "Generate questions covering Python basics: variables, data types, basic operations, simple functions, lists, dictionaries, loops, conditionals.",
                "intermediate": "Generate questions covering Python intermediate topics: OOP concepts, error handling, file I/O, libraries (pandas, numpy), list comprehensions, decorators.",
                "advanced": "Generate questions covering advanced Python: metaclasses, generators, context managers, async programming, optimization, design patterns, testing."
            },
            "sql": {
                "beginner": "Generate questions covering SQL basics: SELECT statements, WHERE clauses, basic joins, GROUP BY, ORDER BY, aggregate functions.",
                "intermediate": "Generate questions covering intermediate SQL: complex joins, subqueries, window functions, CTEs, indexes, data types, constraints.",
                "advanced": "Generate questions covering advanced SQL: query optimization, stored procedures, triggers, partitioning, performance tuning, complex analytics."
            },
            "statistics": {
                "beginner": "Generate questions covering basic statistics: descriptive statistics, mean/median/mode, standard deviation, basic probability.",
                "intermediate": "Generate questions covering intermediate statistics: hypothesis testing, confidence intervals, t-tests, chi-square tests, regression basics.",
                "advanced": "Generate questions covering advanced statistics: multivariate analysis, ANOVA, non-parametric tests, Bayesian statistics, experimental design."
            },
            "machine_learning": {
                "beginner": "Generate questions covering ML basics: supervised vs unsupervised learning, regression vs classification, train/test split, overfitting.",
                "intermediate": "Generate questions covering intermediate ML: feature engineering, cross-validation, model evaluation metrics, ensemble methods, hyperparameter tuning.",
                "advanced": "Generate questions covering advanced ML: deep learning, neural networks, optimization algorithms, regularization techniques, model interpretability."
            },
            "data_visualization": {
                "beginner": "Generate questions covering basic data visualization: chart types, when to use different plots, basic matplotlib/seaborn usage.",
                "intermediate": "Generate questions covering intermediate visualization: interactive plots, dashboard design, color theory, statistical visualizations.",
                "advanced": "Generate questions covering advanced visualization: custom visualizations, D3.js concepts, performance optimization, accessibility, advanced plotly."
            },
            "data_engineering": {
                "beginner": "Generate questions covering basic data engineering: data pipelines, ETL concepts, data formats (CSV, JSON), basic database operations.",
                "intermediate": "Generate questions covering intermediate data engineering: Apache Spark basics, data warehousing, streaming data, data quality, scheduling.",
                "advanced": "Generate questions covering advanced data engineering: distributed systems, real-time processing, data architecture, cloud platforms, optimization."
            }
        }
        
        category_desc = category_prompts.get(request.category, {}).get(request.difficulty, "")
        
        instruction = f"""You are an expert technical interviewer. Generate {request.num_questions} multiple-choice questions for {request.category} assessment at {request.difficulty} level.

{category_desc}

For each question, provide:
1. A clear, challenging question
2. Four plausible options (A, B, C, D)
3. The correct answer index (0-3)
4. A detailed explanation of why the answer is correct
5. Question type: "multiple_choice", "code_analysis", or "scenario"

Include a mix of conceptual and practical questions. For code_analysis questions, provide code snippets.
For scenario questions, present real-world situations.

Return the response in this exact JSON format:
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Detailed explanation here",
            "type": "multiple_choice"
        }}
    ]
}}"""

        response = model.generate_content(instruction)
        response_text = response.text.strip()
        
        # Clean up markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        import json
        try:
            questions_data = json.loads(response_text)
            questions = [Question(**q) for q in questions_data["questions"]]
        except (json.JSONDecodeError, KeyError):
            # Fallback to creating sample questions if parsing fails
            questions = [
                Question(
                    question=f"Sample {request.category} question for {request.difficulty} level",
                    options=["Option A", "Option B", "Option C", "Option D"],
                    correct_answer=0,
                    explanation="This is a sample explanation.",
                    type="multiple_choice"
                )
            ] * request.num_questions
        
        duration_map = {"beginner": 2, "intermediate": 3, "advanced": 4}
        duration = duration_map.get(request.difficulty, 3) * request.num_questions
        
        return SkillAssessmentResponse(
            questions=questions,
            category=request.category,
            difficulty=request.difficulty,
            duration_minutes=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skill-assessment/evaluate", response_model=EvaluationResponse)
async def evaluate_skill_assessment(request: EvaluationRequest):
    """
    Evaluate user's answers and provide detailed feedback
    """
    try:
        # Calculate score
        correct_count = sum(1 for i, answer in enumerate(request.answers) 
                          if i < len(request.questions) and answer == request.questions[i].correct_answer)
        total_questions = len(request.questions)
        score = int((correct_count / total_questions) * 100) if total_questions > 0 else 0
        
        # Determine performance level
        if score >= 90:
            performance_level = "expert"
        elif score >= 75:
            performance_level = "advanced"
        elif score >= 60:
            performance_level = "intermediate"
        else:
            performance_level = "beginner"
        
        # Generate detailed feedback using AI
        feedback_prompt = f"""You are an expert technical mentor. Analyze this skill assessment performance:

Category: {request.category}
Difficulty: {request.difficulty}
Score: {score}% ({correct_count}/{total_questions} correct)
Performance Level: {performance_level}

Provide specific feedback in the following format:

STRENGTHS: (List 2-3 specific strengths based on performance)
WEAKNESSES: (List 2-3 specific areas for improvement)
RECOMMENDATIONS: (List 3-4 specific actionable recommendations for improvement)
DETAILED_FEEDBACK: (Provide encouraging, constructive paragraph about overall performance and next steps)

Base your feedback on the {request.category} domain and {request.difficulty} difficulty level."""

        response = model.generate_content(feedback_prompt)
        feedback_text = response.text.strip()
        
        # Parse the AI feedback (simplified parsing)
        strengths = ["Strong foundation in core concepts", "Good problem-solving approach"]
        weaknesses = ["Could improve in advanced topics", "Practice more complex scenarios"]
        recommendations = [
            "Practice more hands-on exercises",
            "Review fundamental concepts",
            "Work on real-world projects",
            "Join study groups or online communities"
        ]
        
        # Try to extract from AI response if possible
        if "STRENGTHS:" in feedback_text:
            try:
                strengths_section = feedback_text.split("STRENGTHS:")[1].split("WEAKNESSES:")[0].strip()
                strengths = [s.strip("- ").strip() for s in strengths_section.split("\n") if s.strip()]
            except:
                pass
        
        if "WEAKNESSES:" in feedback_text:
            try:
                weaknesses_section = feedback_text.split("WEAKNESSES:")[1].split("RECOMMENDATIONS:")[0].strip()
                weaknesses = [w.strip("- ").strip() for w in weaknesses_section.split("\n") if w.strip()]
            except:
                pass
        
        if "RECOMMENDATIONS:" in feedback_text:
            try:
                recommendations_section = feedback_text.split("RECOMMENDATIONS:")[1].split("DETAILED_FEEDBACK:")[0].strip()
                recommendations = [r.strip("- ").strip() for r in recommendations_section.split("\n") if r.strip()]
            except:
                pass
        
        detailed_feedback = feedback_text if "DETAILED_FEEDBACK:" not in feedback_text else feedback_text.split("DETAILED_FEEDBACK:")[1].strip()
        
        return EvaluationResponse(
            score=score,
            correct_count=correct_count,
            total_questions=total_questions,
            performance_level=performance_level,
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            recommendations=recommendations[:4],
            detailed_feedback=detailed_feedback
        )
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


