import pytest
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Import the app
from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_gemini_model():
    """Mock the Gemini AI model"""
    with patch('main.model') as mock_model:
        mock_response = Mock()
        mock_response.text = "Mocked AI response"
        mock_model.generate_content.return_value = mock_response
        yield mock_model


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing"""
    return [
        {"role": "user", "text": "Hello"},
        {"role": "assistant", "text": "Hi there!"},
        {"role": "user", "text": "How are you?"}
    ]


@pytest.fixture
def sample_prompt_request():
    """Sample prompt request for testing"""
    return {"prompt": "Write a Python function to calculate fibonacci numbers"}


@pytest.fixture
def sample_sql_request():
    """Sample SQL conversion request for testing"""
    return {"english_query": "Show all users who registered in the last month"}


@pytest.fixture
def sample_data_exploration_request():
    """Sample data exploration request for testing"""
    return {
        "description": "E-commerce dataset with customer_id, product_name, purchase_date, amount, category",
        "analysis_type": "eda"
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables"""
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test-api-key-12345'
    }):
        yield


@pytest.fixture(autouse=True)
def setup_test_environment(mock_env_vars):
    """Automatically set up test environment for all tests"""
    pass


class MockGeminiResponse:
    """Helper class to create mock AI responses"""
    
    def __init__(self, text: str):
        self.text = text
    
    def strip(self):
        return self.text.strip()


@pytest.fixture
def create_mock_response():
    """Factory fixture to create mock AI responses"""
    def _create_response(text: str) -> MockGeminiResponse:
        return MockGeminiResponse(text)
    return _create_response
