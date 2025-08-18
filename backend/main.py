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


