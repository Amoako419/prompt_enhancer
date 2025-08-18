import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the app and models
from main import (
    app, 
    PromptRequest, 
    PromptResponse,
    ChatMessage,
    ChatRequest, 
    ChatResponse,
    SqlRequest,
    SqlResponse,
    DataExplorationRequest,
    DataExplorationResponse,
    enhance_prompt,
    chat_endpoint,
    convert_to_sql,
    generate_data_exploration
)

# Create test client
client = TestClient(app)

class TestModels:
    """Test Pydantic models validation"""
    
    def test_prompt_request_model(self):
        """Test PromptRequest model validation"""
        # Valid request
        request = PromptRequest(prompt="Test prompt")
        assert request.prompt == "Test prompt"
        
        # Test with empty string (should be valid)
        request = PromptRequest(prompt="")
        assert request.prompt == ""
    
    def test_prompt_response_model(self):
        """Test PromptResponse model validation"""
        response = PromptResponse(enhanced_prompt="Enhanced test prompt")
        assert response.enhanced_prompt == "Enhanced test prompt"
    
    def test_chat_message_model(self):
        """Test ChatMessage model validation"""
        message = ChatMessage(role="user", text="Hello")
        assert message.role == "user"
        assert message.text == "Hello"
        
        # Test assistant role
        message = ChatMessage(role="assistant", text="Hi there!")
        assert message.role == "assistant"
        assert message.text == "Hi there!"
    
    def test_chat_request_model(self):
        """Test ChatRequest model validation"""
        messages = [
            ChatMessage(role="user", text="Hello"),
            ChatMessage(role="assistant", text="Hi!")
        ]
        request = ChatRequest(history=messages)
        assert len(request.history) == 2
        assert request.history[0].role == "user"
        assert request.history[1].role == "assistant"
    
    def test_chat_response_model(self):
        """Test ChatResponse model validation"""
        response = ChatResponse(reply="Test response")
        assert response.reply == "Test response"
    
    def test_sql_request_model(self):
        """Test SqlRequest model validation"""
        request = SqlRequest(english_query="Show all users")
        assert request.english_query == "Show all users"
    
    def test_sql_response_model(self):
        """Test SqlResponse model validation"""
        response = SqlResponse(sql_query="SELECT * FROM users;")
        assert response.sql_query == "SELECT * FROM users;"
    
    def test_data_exploration_request_model(self):
        """Test DataExplorationRequest model validation"""
        request = DataExplorationRequest(
            description="Test dataset with user data",
            analysis_type="eda"
        )
        assert request.description == "Test dataset with user data"
        assert request.analysis_type == "eda"
    
    def test_data_exploration_response_model(self):
        """Test DataExplorationResponse model validation"""
        response = DataExplorationResponse(code="import pandas as pd")
        assert response.code == "import pandas as pd"


class TestEndpoints:
    """Test API endpoints"""
    
    @patch('main.model')
    def test_enhance_prompt_success(self, mock_model):
        """Test successful prompt enhancement"""
        # Mock the AI response
        mock_response = Mock()
        mock_response.text = "This is an enhanced prompt"
        mock_model.generate_content.return_value = mock_response
        
        # Test the endpoint
        response = client.post("/enhance", json={"prompt": "test prompt"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["enhanced_prompt"] == "This is an enhanced prompt"
        
        # Verify the model was called correctly
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        assert "test prompt" in call_args
        assert "prompt enhancer" in call_args.lower()
    
    @patch('main.model')
    def test_enhance_prompt_with_whitespace(self, mock_model):
        """Test prompt enhancement with whitespace handling"""
        mock_response = Mock()
        mock_response.text = "  Enhanced prompt with spaces  "
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/enhance", json={"prompt": "test"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["enhanced_prompt"] == "Enhanced prompt with spaces"
    
    @patch('main.model')
    def test_enhance_prompt_empty_input(self, mock_model):
        """Test prompt enhancement with empty input"""
        mock_response = Mock()
        mock_response.text = "Please provide a prompt to enhance"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/enhance", json={"prompt": ""})
        
        assert response.status_code == 200
        mock_model.generate_content.assert_called_once()
    
    @patch('main.model')
    def test_enhance_prompt_ai_error(self, mock_model):
        """Test prompt enhancement when AI service fails"""
        mock_model.generate_content.side_effect = Exception("AI service error")
        
        response = client.post("/enhance", json={"prompt": "test"})
        
        assert response.status_code == 500
        data = response.json()
        assert "AI service error" in data["detail"]
    
    def test_enhance_prompt_invalid_input(self):
        """Test prompt enhancement with invalid input"""
        response = client.post("/enhance", json={"invalid_field": "test"})
        assert response.status_code == 422  # Validation error
    
    @patch('main.model')
    def test_chat_endpoint_success(self, mock_model):
        """Test successful chat interaction"""
        mock_response = Mock()
        mock_response.text = "Hello! How can I help you?"
        mock_model.generate_content.return_value = mock_response
        
        chat_data = {
            "history": [
                {"role": "user", "text": "Hello"},
                {"role": "assistant", "text": "Hi there!"},
                {"role": "user", "text": "How are you?"}
            ]
        }
        
        response = client.post("/chat", json=chat_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Hello! How can I help you?"
        
        # Verify the prompt formatting
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        assert "User: Hello" in call_args
        assert "Assistant: Hi there!" in call_args
        assert "User: How are you?" in call_args
        assert call_args.endswith("Assistant:")
    
    @patch('main.model')
    def test_chat_endpoint_empty_history(self, mock_model):
        """Test chat endpoint with empty history"""
        mock_response = Mock()
        mock_response.text = "Hello!"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/chat", json={"history": []})
        
        assert response.status_code == 200
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        assert call_args == "Assistant:"
    
    @patch('main.model')
    def test_chat_endpoint_ai_error(self, mock_model):
        """Test chat endpoint when AI service fails"""
        mock_model.generate_content.side_effect = Exception("Connection timeout")
        
        chat_data = {"history": [{"role": "user", "text": "Hello"}]}
        response = client.post("/chat", json=chat_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Connection timeout" in data["detail"]
    
    @patch('main.model')
    def test_sql_conversion_success(self, mock_model):
        """Test successful SQL conversion"""
        mock_response = Mock()
        mock_response.text = "SELECT * FROM users WHERE age > 18;"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/to-sql", json={"english_query": "Show all adult users"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["sql_query"] == "SELECT * FROM users WHERE age > 18;"
        
        # Verify the prompt includes SQL instruction
        call_args = mock_model.generate_content.call_args[0][0]
        assert "SQL generator" in call_args
        assert "Show all adult users" in call_args
    
    @patch('main.model')
    def test_sql_conversion_with_markdown_cleanup(self, mock_model):
        """Test SQL conversion with markdown formatting cleanup"""
        mock_response = Mock()
        mock_response.text = "```sql\nSELECT * FROM products;\n```"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/to-sql", json={"english_query": "Show all products"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["sql_query"] == "SELECT * FROM products;"
    
    @patch('main.model')
    def test_sql_conversion_with_simple_markdown(self, mock_model):
        """Test SQL conversion with simple markdown cleanup"""
        mock_response = Mock()
        mock_response.text = "```\nSELECT name FROM customers;\n```"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/to-sql", json={"english_query": "Get customer names"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["sql_query"] == "SELECT name FROM customers;"
    
    @patch('main.model')
    def test_data_exploration_eda_success(self, mock_model):
        """Test successful EDA code generation"""
        mock_response = Mock()
        mock_response.text = "import pandas as pd\ndf.info()\ndf.describe()"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "Customer dataset with age, income, purchases",
            "analysis_type": "eda"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "import pandas as pd" in data["code"]
        
        # Verify EDA-specific prompt
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Exploratory Data Analysis" in call_args
        assert "Customer dataset with age, income, purchases" in call_args
    
    @patch('main.model')
    def test_data_exploration_statistical_analysis(self, mock_model):
        """Test statistical analysis code generation"""
        mock_response = Mock()
        mock_response.text = "from scipy import stats\nstats.ttest_ind(group1, group2)"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "A/B test data",
            "analysis_type": "statistical"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "scipy" in data["code"]
        
        call_args = mock_model.generate_content.call_args[0][0]
        assert "statistician" in call_args
        assert "hypothesis testing" in call_args
    
    @patch('main.model')
    def test_data_exploration_anomaly_detection(self, mock_model):
        """Test anomaly detection code generation"""
        mock_response = Mock()
        mock_response.text = "from sklearn.ensemble import IsolationForest\nmodel.fit(X)"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "Network traffic data",
            "analysis_type": "anomaly"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "sklearn" in data["code"]
        
        call_args = mock_model.generate_content.call_args[0][0]
        assert "anomaly detection" in call_args
        assert "outlier detection" in call_args
    
    @patch('main.model')
    def test_data_exploration_timeseries(self, mock_model):
        """Test time series analysis code generation"""
        mock_response = Mock()
        mock_response.text = "import statsmodels.api as sm\ndecomposition = sm.tsa.seasonal_decompose(ts)"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "Daily sales data",
            "analysis_type": "timeseries"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "statsmodels" in data["code"]
        
        call_args = mock_model.generate_content.call_args[0][0]
        assert "time series" in call_args
        assert "trend analysis" in call_args
    
    @patch('main.model')
    def test_data_exploration_visualization(self, mock_model):
        """Test visualization code generation"""
        mock_response = Mock()
        mock_response.text = "import matplotlib.pyplot as plt\nimport seaborn as sns"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "Survey responses",
            "analysis_type": "visualization"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "matplotlib" in data["code"]
        
        call_args = mock_model.generate_content.call_args[0][0]
        assert "visualization" in call_args
        assert "matplotlib" in call_args
    
    @patch('main.model')
    def test_data_exploration_unknown_type_defaults_to_eda(self, mock_model):
        """Test that unknown analysis type defaults to EDA"""
        mock_response = Mock()
        mock_response.text = "import pandas as pd"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "Some dataset",
            "analysis_type": "unknown_type"
        })
        
        assert response.status_code == 200
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Exploratory Data Analysis" in call_args  # Should default to EDA
    
    @patch('main.model')
    def test_data_exploration_python_markdown_cleanup(self, mock_model):
        """Test data exploration with Python markdown cleanup"""
        mock_response = Mock()
        mock_response.text = "```python\nimport pandas as pd\nprint('Hello')\n```"
        mock_model.generate_content.return_value = mock_response
        
        response = client.post("/data-exploration", json={
            "description": "Test dataset",
            "analysis_type": "eda"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "import pandas as pd\nprint('Hello')"


class TestAsyncFunctions:
    """Test async endpoint functions directly"""
    
    @pytest.mark.asyncio
    @patch('main.model')
    async def test_enhance_prompt_async(self, mock_model):
        """Test enhance_prompt function directly"""
        mock_response = Mock()
        mock_response.text = "Enhanced async prompt"
        mock_model.generate_content.return_value = mock_response
        
        request = PromptRequest(prompt="test prompt")
        result = await enhance_prompt(request)
        
        assert isinstance(result, PromptResponse)
        assert result.enhanced_prompt == "Enhanced async prompt"
    
    @pytest.mark.asyncio
    @patch('main.model')
    async def test_enhance_prompt_async_exception(self, mock_model):
        """Test enhance_prompt exception handling"""
        mock_model.generate_content.side_effect = Exception("Async error")
        
        request = PromptRequest(prompt="test")
        
        with pytest.raises(HTTPException) as exc_info:
            await enhance_prompt(request)
        
        assert exc_info.value.status_code == 500
        assert "Async error" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch('main.model')
    async def test_chat_endpoint_async(self, mock_model):
        """Test chat_endpoint function directly"""
        mock_response = Mock()
        mock_response.text = "Async chat response"
        mock_model.generate_content.return_value = mock_response
        
        messages = [ChatMessage(role="user", text="Hello")]
        request = ChatRequest(history=messages)
        result = await chat_endpoint(request)
        
        assert isinstance(result, ChatResponse)
        assert result.reply == "Async chat response"
    
    @pytest.mark.asyncio
    @patch('main.model')
    async def test_convert_to_sql_async(self, mock_model):
        """Test convert_to_sql function directly"""
        mock_response = Mock()
        mock_response.text = "SELECT * FROM async_table;"
        mock_model.generate_content.return_value = mock_response
        
        request = SqlRequest(english_query="Show async data")
        result = await convert_to_sql(request)
        
        assert isinstance(result, SqlResponse)
        assert result.sql_query == "SELECT * FROM async_table;"
    
    @pytest.mark.asyncio
    @patch('main.model')
    async def test_generate_data_exploration_async(self, mock_model):
        """Test generate_data_exploration function directly"""
        mock_response = Mock()
        mock_response.text = "import asyncio\nprint('Async EDA')"
        mock_model.generate_content.return_value = mock_response
        
        request = DataExplorationRequest(
            description="Async dataset",
            analysis_type="eda"
        )
        result = await generate_data_exploration(request)
        
        assert isinstance(result, DataExplorationResponse)
        assert "import asyncio" in result.code


class TestIntegration:
    """Integration tests for multiple endpoints"""
    
    @patch('main.model')
    def test_full_workflow_integration(self, mock_model):
        """Test a full workflow using multiple endpoints"""
        # Mock different responses for different calls
        mock_responses = [
            Mock(text="Enhanced: Analyze customer data"),  # enhance
            Mock(text="SELECT * FROM customers;"),         # to-sql
            Mock(text="import pandas as pd\ndf.head()")   # data-exploration
        ]
        mock_model.generate_content.side_effect = mock_responses
        
        # Step 1: Enhance a prompt
        response1 = client.post("/enhance", json={
            "prompt": "Analyze customer data"
        })
        assert response1.status_code == 200
        
        # Step 2: Convert to SQL
        response2 = client.post("/to-sql", json={
            "english_query": "Show all customers"
        })
        assert response2.status_code == 200
        
        # Step 3: Generate analysis code
        response3 = client.post("/data-exploration", json={
            "description": "Customer dataset analysis",
            "analysis_type": "eda"
        })
        assert response3.status_code == 200
        
        # Verify all calls were made
        assert mock_model.generate_content.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
