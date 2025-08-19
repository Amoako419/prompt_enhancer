from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
import re
import json
import pandas as pd
import numpy as np
import io
import subprocess
import tempfile

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Utility function to convert numpy types to native Python types
def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # For numpy scalars
        return obj.item()
    else:
        return obj
model = genai.GenerativeModel("gemini-2.0-flash")

# Text formatting utilities
def clean_llm_text(text):
    """Clean and format text from LLM responses"""
    if not text:
        return ""
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code markers
    text = re.sub(r'#{1,6}\s+', '', text)         # Remove headers
    
    # Remove common prefixes
    prefixes = [
        r'^(Answer:|Question:|Explanation:|Note:|STRENGTHS:|WEAKNESSES:|RECOMMENDATIONS:|DETAILED_FEEDBACK:)',
        r'^\s*[-*+]\s+',  # Bullet points
        r'^\s*\d+\.\s+',  # Numbered lists
        r'^[A-D]\)\s*',   # Option letters
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up whitespace and punctuation
    text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
    text = re.sub(r'([.!?])\s*([.!?])', r'\1', text)  # Remove duplicate punctuation
    
    return text.strip()

def format_question_data(questions_data):
    """Format question data from LLM response"""
    formatted_questions = []
    
    for q in questions_data.get("questions", []):
        formatted_q = {
            "question": clean_llm_text(q.get("question", "")),
            "options": [clean_llm_text(opt) for opt in q.get("options", [])],
            "correct_answer": q.get("correct_answer", 0),
            "explanation": clean_llm_text(q.get("explanation", "")),
            "type": q.get("type", "multiple_choice")
        }
        formatted_questions.append(formatted_q)
    
    return formatted_questions

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

class FieldSchema(BaseModel):
    name: str
    type: str
    constraints: str

class SchemaConfig(BaseModel):
    tableName: str
    recordCount: int
    format: str
    fields: List[FieldSchema]

class PipelineConfig(BaseModel):
    testType: str
    dataQuality: str
    includeEdgeCases: bool
    includeNulls: bool
    duplicatePercentage: int

class DataPipelineRequest(BaseModel):
    schema: SchemaConfig
    pipeline: PipelineConfig

class DataPipelineResponse(BaseModel):
    data: str
    preview: str
    metadata: dict

class MCPAnalysisRequest(BaseModel):
    analysis_type: str
    file_data: Optional[str] = None
    parameters: dict = {}

class MCPAnalysisResponse(BaseModel):
    status: str
    results: dict
    visualizations: List[dict] = []

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

IMPORTANT: Return ONLY valid JSON in the exact format below. Do not include any markdown, explanations, or additional text.

{{
    "questions": [
        {{
            "question": "What is the primary purpose of pandas in Python data science?",
            "options": ["Web development", "Data manipulation and analysis", "Game development", "System administration"],
            "correct_answer": 1,
            "explanation": "pandas is specifically designed for data manipulation and analysis tasks in Python.",
            "type": "multiple_choice"
        }}
    ]
}}

Generate {request.num_questions} questions following this exact format. Each question should be:
- Clear and specific to {request.category}
- Appropriate for {request.difficulty} level
- Have 4 realistic options
- Include correct_answer as index (0-3)
- Provide detailed explanation

Return only the JSON object, no additional text or formatting."""

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
            formatted_questions = format_question_data(questions_data)
            questions = [Question(**q) for q in formatted_questions]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            
            # Try to extract questions manually if JSON parsing fails
            questions = []
            if "question" in response_text.lower():
                # Try manual parsing for common AI response patterns
                import re
                
                # Pattern to match question blocks
                question_pattern = r'"question":\s*"([^"]*)".*?"options":\s*\[(.*?)\].*?"correct_answer":\s*(\d+).*?"explanation":\s*"([^"]*)"'
                matches = re.findall(question_pattern, response_text, re.DOTALL)
                
                for match in matches[:request.num_questions]:
                    question_text, options_text, correct_answer, explanation = match
                    # Parse options
                    options_matches = re.findall(r'"([^"]*)"', options_text)
                    if len(options_matches) >= 4:
                        questions.append(Question(
                            question=clean_llm_text(question_text),
                            options=[clean_llm_text(opt) for opt in options_matches[:4]],
                            correct_answer=int(correct_answer),
                            explanation=clean_llm_text(explanation),
                            type="multiple_choice"
                        ))
            
            # If manual parsing also fails, create meaningful fallback questions
            if not questions:
                fallback_questions = {
                    "Python for Data Science": [
                        {
                            "question": "Which library is primarily used for data manipulation in Python?",
                            "options": ["pandas", "matplotlib", "requests", "os"],
                            "correct_answer": 0,
                            "explanation": "pandas is the primary library for data manipulation and analysis in Python.",
                            "type": "multiple_choice"
                        },
                        {
                            "question": "What method is used to read a CSV file in pandas?",
                            "options": ["read_csv()", "load_csv()", "import_csv()", "get_csv()"],
                            "correct_answer": 0,
                            "explanation": "pandas.read_csv() is the standard method to read CSV files into a DataFrame.",
                            "type": "multiple_choice"
                        }
                    ],
                    "SQL and Databases": [
                        {
                            "question": "Which SQL clause is used to filter rows?",
                            "options": ["SELECT", "WHERE", "ORDER BY", "GROUP BY"],
                            "correct_answer": 1,
                            "explanation": "The WHERE clause is used to filter rows based on specified conditions.",
                            "type": "multiple_choice"
                        }
                    ]
                }
                
                category_questions = fallback_questions.get(request.category, fallback_questions["Python for Data Science"])
                questions = [Question(**q) for q in category_questions * (request.num_questions // len(category_questions) + 1)][:request.num_questions]
        
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
                strengths = [clean_llm_text(s.strip("- ").strip()) for s in strengths_section.split("\n") if s.strip()]
            except:
                pass
        
        if "WEAKNESSES:" in feedback_text:
            try:
                weaknesses_section = feedback_text.split("WEAKNESSES:")[1].split("RECOMMENDATIONS:")[0].strip()
                weaknesses = [clean_llm_text(w.strip("- ").strip()) for w in weaknesses_section.split("\n") if w.strip()]
            except:
                pass
        
        if "RECOMMENDATIONS:" in feedback_text:
            try:
                recommendations_section = feedback_text.split("RECOMMENDATIONS:")[1].split("DETAILED_FEEDBACK:")[0].strip()
                recommendations = [clean_llm_text(r.strip("- ").strip()) for r in recommendations_section.split("\n") if r.strip()]
            except:
                pass
        
        detailed_feedback = feedback_text if "DETAILED_FEEDBACK:" not in feedback_text else feedback_text.split("DETAILED_FEEDBACK:")[1].strip()
        detailed_feedback = clean_llm_text(detailed_feedback)
        
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

@app.post("/generate-pipeline-data", response_model=DataPipelineResponse)
async def generate_pipeline_data(req: DataPipelineRequest):
    """
    Generate test data for data pipeline testing based on schema and configuration
    """
    try:
        import csv
        import json
        import random
        import string
        from datetime import datetime, timedelta
        from faker import Faker
        
        fake = Faker()
        
        # Create AI prompt for realistic data generation
        prompt = f"""
        Generate realistic test data for a data pipeline with the following specifications:
        
        Table: {req.schema.tableName}
        Record Count: {req.schema.recordCount}
        Test Type: {req.pipeline.testType}
        Data Quality: {req.pipeline.dataQuality}
        Include Edge Cases: {req.pipeline.includeEdgeCases}
        Include Nulls: {req.pipeline.includeNulls}
        Duplicate Percentage: {req.pipeline.duplicatePercentage}%
        
        Schema Fields:
        {[f"{field.name} ({field.type}) - {field.constraints}" for field in req.schema.fields]}
        
        Please provide guidance on generating realistic data that matches this schema.
        Focus on data types, constraints, and testing requirements.
        """
        
        # Get AI guidance for data generation
        response = model.generate_content(prompt)
        ai_guidance = response.text
        
        # Generate actual data based on schema
        generated_records = []
        
        # Helper function to generate data by type
        def generate_value(field_type, constraints, include_null=False):
            if include_null and random.random() < 0.1:  # 10% chance of null
                return None
                
            if field_type == 'integer':
                if 'primary_key' in constraints or 'auto_increment' in constraints:
                    return len(generated_records) + 1
                return random.randint(1, 10000)
            elif field_type == 'string':
                return fake.name() if 'name' in constraints.lower() else fake.word()
            elif field_type == 'email':
                return fake.email()
            elif field_type == 'phone':
                return fake.phone_number()
            elif field_type == 'url':
                return fake.url()
            elif field_type == 'uuid':
                return str(fake.uuid4())
            elif field_type == 'datetime':
                return fake.date_time_between(start_date='-1y', end_date='now').isoformat()
            elif field_type == 'date':
                return fake.date_between(start_date='-1y', end_date='today').isoformat()
            elif field_type == 'boolean':
                return random.choice([True, False])
            elif field_type == 'float':
                return round(random.uniform(0, 1000), 2)
            elif field_type == 'json':
                return json.dumps({"key": fake.word(), "value": fake.sentence()})
            elif field_type == 'text':
                return fake.text(max_nb_chars=200)
            else:
                return fake.word()
        
        # Generate records
        for i in range(req.schema.recordCount):
            record = {}
            for field in req.schema.fields:
                include_null = req.pipeline.includeNulls and 'not_null' not in field.constraints
                value = generate_value(field.type, field.constraints, include_null)
                record[field.name] = value
            generated_records.append(record)
        
        # Add duplicates if requested
        if req.pipeline.duplicatePercentage > 0:
            duplicate_count = int(len(generated_records) * req.pipeline.duplicatePercentage / 100)
            for _ in range(duplicate_count):
                if generated_records:
                    duplicate_record = random.choice(generated_records).copy()
                    generated_records.append(duplicate_record)
        
        # Add edge cases if requested
        if req.pipeline.includeEdgeCases:
            edge_cases = []
            for field in req.schema.fields:
                if field.type == 'string':
                    edge_cases.append({field.name: ""})  # Empty string
                    edge_cases.append({field.name: "a" * 1000})  # Very long string
                elif field.type == 'integer':
                    edge_cases.append({field.name: 0})
                    edge_cases.append({field.name: -1})
                    edge_cases.append({field.name: 999999999})
                elif field.type == 'email':
                    edge_cases.append({field.name: "invalid-email"})
                    edge_cases.append({field.name: "@domain.com"})
            
            # Add a few edge case records
            for i in range(min(5, len(edge_cases))):
                edge_record = {}
                for field in req.schema.fields:
                    edge_record[field.name] = generate_value(field.type, field.constraints)
                
                # Apply one edge case
                if i < len(edge_cases):
                    edge_record.update(edge_cases[i])
                
                generated_records.append(edge_record)
        
        # Format output based on requested format
        if req.schema.format == 'csv':
            output = []
            if generated_records:
                # CSV header
                headers = [field.name for field in req.schema.fields]
                output.append(','.join(headers))
                
                # CSV rows
                for record in generated_records:
                    row = []
                    for field in req.schema.fields:
                        value = record.get(field.name, '')
                        if value is None:
                            row.append('')
                        elif isinstance(value, str):
                            row.append(f'"{value}"')
                        else:
                            row.append(str(value))
                    output.append(','.join(row))
            
            data_content = '\n'.join(output)
            
        elif req.schema.format == 'json':
            data_content = json.dumps(generated_records, indent=2, default=str)
            
        elif req.schema.format == 'sql':
            if generated_records:
                headers = [field.name for field in req.schema.fields]
                insert_statements = []
                insert_statements.append(f"-- Generated test data for {req.schema.tableName}")
                
                for record in generated_records:
                    values = []
                    for field in req.schema.fields:
                        value = record.get(field.name)
                        if value is None:
                            values.append('NULL')
                        elif isinstance(value, str):
                            values.append(f"'{value.replace("'", "''")}'")
                        else:
                            values.append(str(value))
                    
                    insert_sql = f"INSERT INTO {req.schema.tableName} ({', '.join(headers)}) VALUES ({', '.join(values)});"
                    insert_statements.append(insert_sql)
                
                data_content = '\n'.join(insert_statements)
            else:
                data_content = f"-- No data generated for {req.schema.tableName}"
                
        else:  # parquet or other formats
            data_content = json.dumps(generated_records, indent=2, default=str)
        
        # Create preview (first 5 records)
        preview_records = generated_records[:5]
        if req.schema.format == 'csv':
            preview_lines = data_content.split('\n')[:6]  # Header + 5 records
            preview = '\n'.join(preview_lines)
        elif req.schema.format == 'json':
            preview = json.dumps(preview_records, indent=2, default=str)
        elif req.schema.format == 'sql':
            preview_lines = data_content.split('\n')[:6]  # Comment + 5 statements
            preview = '\n'.join(preview_lines)
        else:
            preview = json.dumps(preview_records, indent=2, default=str)
        
        # Calculate metadata
        data_size = len(data_content.encode('utf-8'))
        file_size = f"{data_size / 1024:.1f} KB" if data_size < 1024*1024 else f"{data_size / (1024*1024):.1f} MB"
        
        metadata = {
            "recordCount": len(generated_records),
            "fileSize": file_size,
            "generationTime": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "duplicateCount": int(len(generated_records) * req.pipeline.duplicatePercentage / 100) if req.pipeline.duplicatePercentage > 0 else 0,
            "hasEdgeCases": req.pipeline.includeEdgeCases,
            "hasNulls": req.pipeline.includeNulls,
            "dataQuality": req.pipeline.dataQuality,
            "testType": req.pipeline.testType
        }
        
        return DataPipelineResponse(
            data=data_content,
            preview=preview,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"Pipeline data generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate pipeline data: {str(e)}")

@app.get("/mcp/health")
async def mcp_health_check():
    """
    Check if MCP services are available
    """
    return {"status": "connected", "message": "MCP proxy is running"}

@app.post("/mcp/load-data")
async def mcp_load_data(file: UploadFile = File(...)):
    """
    Load and preview data file
    """
    try:
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        results = {
            "title": "Data Loading Summary",
            "content": f"""📁 File: {file.filename}
📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns

📋 Column Information:
{chr(10).join([f"  • {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items()])}

🔍 Data Preview (First 5 rows):
{df.head().to_string()}

⚠️ Missing Values:
{chr(10).join([f"  • {col}: {count} missing" for col, count in df.isnull().sum().items() if count > 0]) or "  • No missing values found"}

💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB""",
            "data": convert_numpy_types({
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": df.head().to_dict()
            })
        }
        
        return MCPAnalysisResponse(
            status="success",
            results=results,
            visualizations=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

@app.post("/mcp/descriptive-stats")
async def mcp_descriptive_stats(file: UploadFile = File(...)):
    """
    Generate descriptive statistics for numeric columns
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            stats_summary = numeric_df.describe()
            
            formatted_stats = []
            for col in stats_summary.columns:
                col_stats = stats_summary[col]
                formatted_stats.append(f"""
📊 {col}:
  • Count: {col_stats['count']:.0f} values
  • Mean: {col_stats['mean']:.2f}
  • Median (50%): {col_stats['50%']:.2f}
  • Std Dev: {col_stats['std']:.2f}
  • Min: {col_stats['min']:.2f}
  • Max: {col_stats['max']:.2f}
  • Range: {col_stats['max'] - col_stats['min']:.2f}""")
            
            results = {
                "title": "Descriptive Statistics Summary",
                "content": f"""📈 Statistical Analysis for {len(numeric_df.columns)} numeric columns:

{chr(10).join(formatted_stats)}

📋 Summary:
  • Total numeric columns: {len(numeric_df.columns)}
  • Total records: {len(df)}
  • Columns analyzed: {', '.join(numeric_df.columns.tolist())}""",
                "data": convert_numpy_types({
                    "statistics": stats_summary.to_dict(),
                    "numeric_columns": numeric_df.columns.tolist()
                })
            }
        else:
            results = {
                "title": "No Numeric Data Found",
                "content": "❌ Error: No numeric columns found for statistical analysis.\n\nAvailable columns:\n" + 
                         "\n".join([f"  • {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items()]),
                "error": True
            }
        
        return MCPAnalysisResponse(
            status="success",
            results=results,
            visualizations=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate statistics: {str(e)}")

@app.post("/mcp/correlation-analysis")
async def mcp_correlation_analysis(file: UploadFile = File(...)):
    """
    Perform correlation analysis on numeric columns
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strength = "Very Strong" if abs(corr_val) > 0.8 else "Strong" if abs(corr_val) > 0.6 else "Moderate"
                        direction = "Positive" if corr_val > 0 else "Negative"
                        strong_correlations.append(f"  • {col1} ↔ {col2}: {corr_val:.3f} ({strength} {direction})")
            
            # Format correlation matrix for display
            corr_display = []
            for col in corr_matrix.columns:
                row_values = [f"{corr_matrix.loc[col, other_col]:.3f}" for other_col in corr_matrix.columns]
                corr_display.append(f"  {col:<15} | " + " | ".join(f"{val:>8}" for val in row_values))
            
            results = {
                "title": "Correlation Analysis Results",
                "content": f"""🔗 Correlation Analysis for {len(numeric_df.columns)} numeric variables:

📊 Correlation Matrix:
  {'Variable':<15} | {' | '.join(f'{col:>8}' for col in corr_matrix.columns)}
  {'-' * (15 + len(corr_matrix.columns) * 11)}
{chr(10).join(corr_display)}

🎯 Notable Correlations (|r| > 0.5):
{chr(10).join(strong_correlations) if strong_correlations else "  • No strong correlations found (all |r| ≤ 0.5)"}

💡 Interpretation Guide:
  • |r| > 0.8: Very strong correlation
  • |r| > 0.6: Strong correlation  
  • |r| > 0.3: Moderate correlation
  • |r| ≤ 0.3: Weak correlation""",
                "data": convert_numpy_types({
                    "correlation_matrix": corr_matrix.to_dict(),
                    "strong_correlations": [
                        {"variables": [corr_matrix.columns[i], corr_matrix.columns[j]], 
                         "correlation": float(corr_matrix.iloc[i, j])}
                        for i in range(len(corr_matrix.columns))
                        for j in range(i+1, len(corr_matrix.columns))
                        if abs(corr_matrix.iloc[i, j]) > 0.5
                    ]
                })
            }
        else:
            results = {
                "title": "Insufficient Data for Correlation",
                "content": f"""❌ Error: Need at least 2 numeric columns for correlation analysis.

Available columns:
{chr(10).join([f"  • {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items()])}

Numeric columns found: {len(numeric_df.columns)}""",
                "error": True
            }
        
        return MCPAnalysisResponse(
            status="success",
            results=results,
            visualizations=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform correlation analysis: {str(e)}")

@app.post("/mcp/visualization")
async def mcp_visualization(file: UploadFile = File(...)):
    """
    Generate data visualizations
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        visualizations = []
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                n_plots = min(3, len(numeric_cols))
                fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
                if n_plots == 1:
                    axes = [axes]
                
                for i, col in enumerate(numeric_cols[:n_plots]):
                    ax = axes[i] if n_plots > 1 else axes[0]
                    ax.hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                visualizations.append({
                    "title": "Data Distribution Histograms",
                    "image": image_base64,
                    "type": "histogram"
                })
                
                results = {
                    "title": "Data Visualization Generated",
                    "content": f"""📊 Visualization Summary:

✅ Successfully created distribution plots for {n_plots} numeric columns:
{chr(10).join([f"  • {col}: {len(df[col].dropna())} values plotted" for col in numeric_cols[:n_plots]])}

📈 Chart Details:
  • Chart Type: Histograms (20 bins each)
  • Columns Visualized: {', '.join(numeric_cols[:n_plots])}
  • Missing Values: Excluded from plots
  • Image Format: PNG (high resolution)

💡 Insights:
{chr(10).join([f"  • {col}: Range from {df[col].min():.2f} to {df[col].max():.2f}" for col in numeric_cols[:n_plots]])}""",
                    "data": convert_numpy_types({
                        "columns_plotted": numeric_cols[:n_plots].tolist(),
                        "total_numeric_columns": len(numeric_cols)
                    })
                }
            else:
                results = {
                    "title": "No Numeric Data for Visualization",
                    "content": f"""❌ Error: No numeric columns found for visualization.

Available columns:
{chr(10).join([f"  • {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items()])}

💡 Suggestion: Ensure your dataset contains numeric columns for visualization.""",
                    "error": True
                }
                
        except ImportError as e:
            results = {
                "title": "Visualization Library Error",
                "content": f"""❌ Error: Required visualization libraries not available.
                
Error details: {str(e)}

💡 Solution: Install required packages:
  pip install matplotlib seaborn""",
                "error": True
            }
        
        return MCPAnalysisResponse(
            status="success",
            results=results,
            visualizations=visualizations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate visualizations: {str(e)}")

@app.post("/mcp/hypothesis-testing")
async def mcp_hypothesis_testing(file: UploadFile = File(...)):
    """
    Perform hypothesis testing (normality tests)
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 1:
            try:
                from scipy import stats
                
                test_results = []
                for col in numeric_cols[:3]:
                    data = df[col].dropna()
                    
                    if len(data) > 3:
                        stat, p_value = stats.shapiro(data)
                        interpretation = "normally distributed" if p_value > 0.05 else "not normally distributed"
                        
                        test_results.append(f"""
📊 {col}:
  • Sample size: {len(data)} values
  • Test statistic: {stat:.4f}
  • P-value: {p_value:.6f}
  • Result: Data appears {interpretation} (α = 0.05)
  • Mean: {data.mean():.3f}
  • Std Dev: {data.std():.3f}""")
                    else:
                        test_results.append(f"""
📊 {col}:
  • Sample size: {len(data)} values
  • Status: ❌ Insufficient data (need > 3 values)""")
                
                results = {
                    "title": "Hypothesis Testing Results",
                    "content": f"""🧪 Shapiro-Wilk Normality Test Results:

{chr(10).join(test_results)}

📋 Test Summary:
  • Test Type: Shapiro-Wilk Normality Test
  • Null Hypothesis (H₀): Data is normally distributed
  • Alternative Hypothesis (H₁): Data is not normally distributed
  • Significance Level: α = 0.05

💡 Interpretation:
  • P-value > 0.05: Fail to reject H₀ (data appears normal)
  • P-value ≤ 0.05: Reject H₀ (data appears non-normal)
  
🔬 Columns tested: {', '.join(numeric_cols[:3])}""",
                    "data": convert_numpy_types({
                        "test_results": [
                            {
                                "column": col,
                                "statistic": float(stats.shapiro(df[col].dropna())[0]) if len(df[col].dropna()) > 3 else None,
                                "p_value": float(stats.shapiro(df[col].dropna())[1]) if len(df[col].dropna()) > 3 else None,
                                "sample_size": len(df[col].dropna()),
                                "is_normal": bool(stats.shapiro(df[col].dropna())[1] > 0.05) if len(df[col].dropna()) > 3 else None
                            }
                            for col in numeric_cols[:3]
                        ]
                    })
                }
                
            except ImportError:
                results = {
                    "title": "Statistical Library Missing",
                    "content": """❌ Error: SciPy library not available for statistical tests.

💡 Solution: Install required package:
  pip install scipy

🔬 Available Tests (when SciPy is installed):
  • Shapiro-Wilk Normality Test
  • T-tests (one-sample, two-sample)
  • Chi-square tests
  • ANOVA tests""",
                    "error": True
                }
        else:
            results = {
                "title": "No Numeric Data for Testing",
                "content": f"""❌ Error: No numeric columns found for hypothesis testing.

Available columns:
{chr(10).join([f"  • {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items()])}

💡 Suggestion: Ensure your dataset contains numeric columns for statistical testing.""",
                "error": True
            }
        
        return MCPAnalysisResponse(
            status="success",
            results=results,
            visualizations=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform hypothesis testing: {str(e)}")

@app.post("/mcp/machine-learning")
async def mcp_machine_learning(file: UploadFile = File(...)):
    """
    Perform machine learning analysis (K-means clustering)
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                import numpy as np
                
                features = df[numeric_cols].dropna()
                if len(features) >= 3:
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    n_clusters = min(4, max(2, len(features) // 10))
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    cluster_info = []
                    for i in range(n_clusters):
                        cluster_mask = clusters == i
                        cluster_size = sum(cluster_mask)
                        cluster_data = features[cluster_mask]
                        
                        cluster_info.append(f"""
🎯 Cluster {i+1}:
  • Size: {cluster_size} samples ({cluster_size/len(features)*100:.1f}%)
  • Characteristics:
{chr(10).join([f"    - {col}: avg = {cluster_data[col].mean():.2f}" for col in numeric_cols[:3]])}""")
                    
                    results = {
                        "title": "K-Means Clustering Analysis",
                        "content": f"""🤖 Machine Learning Analysis Results:

📊 Clustering Summary:
  • Algorithm: K-Means Clustering
  • Number of clusters: {n_clusters}
  • Features used: {', '.join(numeric_cols.tolist())}
  • Samples analyzed: {len(features)}
  • Inertia (sum of squared distances): {kmeans.inertia_:.2f}

{chr(10).join(cluster_info)}

🔬 Technical Details:
  • Data preprocessing: StandardScaler (mean=0, std=1)
  • Initialization: k-means++ (smart centroid selection)
  • Random state: 42 (reproducible results)
  • Convergence: {kmeans.n_iter_} iterations

💡 Insights:
  • Clusters represent groups of similar data points
  • Lower inertia indicates better cluster cohesion
  • Each cluster has distinct characteristics in the features""",
                        "data": convert_numpy_types({
                            "model_type": "K-Means Clustering",
                            "n_clusters": n_clusters,
                            "inertia": float(kmeans.inertia_),
                            "features_used": numeric_cols.tolist(),
                            "cluster_distribution": {
                                f"Cluster {i+1}": int(sum(clusters == i))
                                for i in range(n_clusters)
                            },
                            "n_iterations": int(kmeans.n_iter_)
                        })
                    }
                else:
                    results = {
                        "title": "Insufficient Data for ML",
                        "content": f"""❌ Error: Insufficient data for machine learning analysis.

Data available:
  • Numeric columns: {len(numeric_cols)}
  • Complete records: {len(features)}
  • Required: At least 3 complete records

💡 Suggestion: Ensure you have more data rows with complete numeric values.""",
                        "error": True
                    }
                    
            except ImportError as e:
                results = {
                    "title": "ML Library Missing",
                    "content": f"""❌ Error: Machine learning libraries not available.

Missing: {str(e)}

💡 Solution: Install required packages:
  pip install scikit-learn numpy

🤖 Available ML Algorithms (when installed):
  • K-Means Clustering
  • Linear/Logistic Regression
  • Random Forest
  • Support Vector Machines""",
                    "error": True
                }
        else:
            results = {
                "title": "Insufficient Features for ML",
                "content": f"""❌ Error: Need at least 2 numeric columns for machine learning.

Available columns:
{chr(10).join([f"  • {col} ({dtype})" for col, dtype in df.dtypes.astype(str).items()])}

Numeric columns found: {len(numeric_cols)}

💡 Suggestion: Ensure your dataset has multiple numeric features for analysis.""",
                "error": True
            }
        
        return MCPAnalysisResponse(
            status="success",
            results=results,
            visualizations=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform machine learning analysis: {str(e)}")

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


