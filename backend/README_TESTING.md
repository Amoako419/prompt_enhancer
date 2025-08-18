# Backend Testing Guide

This directory contains comprehensive unit tests for the Prompt Enhancer backend API.

## 📁 Test Files Structure

```
backend/
├── main.py                 # Main FastAPI application
├── test_main.py           # Comprehensive unit tests
├── conftest.py            # Pytest fixtures and configuration
├── pytest.ini            # Pytest configuration
├── requirements-test.txt  # Testing dependencies
├── run_tests.py          # Test runner script
└── README_TESTING.md     # This file
```

## 🧪 Test Coverage

The test suite covers:

### **Model Validation Tests**
- ✅ Pydantic model validation for all request/response models
- ✅ Data type validation and edge cases
- ✅ Required field validation

### **API Endpoint Tests**
- ✅ **Prompt Enhancement** (`/enhance`)
  - Success scenarios with various inputs
  - Empty prompt handling
  - Whitespace trimming
  - AI service error handling
  - Input validation errors

- ✅ **Chat Functionality** (`/chat`)
  - Multi-message conversations
  - Empty chat history
  - Proper prompt formatting
  - Role-based message handling
  - AI service failures

- ✅ **SQL Conversion** (`/to-sql`)
  - English to SQL translation
  - Markdown formatting cleanup
  - Various SQL query types
  - Error handling

- ✅ **Data Exploration** (`/data-exploration`)
  - All 5 analysis types (EDA, Statistical, Anomaly, TimeSeries, Visualization)
  - Code generation and formatting
  - Python markdown cleanup
  - Unknown analysis type fallback

### **Async Function Tests**
- ✅ Direct testing of async endpoint functions
- ✅ Exception handling in async context
- ✅ Proper response model validation

### **Integration Tests**
- ✅ Multi-endpoint workflow testing
- ✅ Full application flow validation

## 🚀 Running Tests

### Option 1: Using the Test Runner (Recommended)
```bash
python run_tests.py
```

This interactive script will:
1. Install test dependencies automatically
2. Offer options for basic tests or coverage analysis
3. Provide colored output and detailed results

### Option 2: Manual Pytest Execution
```bash
# Install dependencies
pip install -r requirements-test.txt

# Run all tests
pytest test_main.py -v

# Run with coverage
pip install coverage
coverage run -m pytest test_main.py
coverage report -m
```

### Option 3: Run Specific Test Classes
```bash
# Test only model validation
pytest test_main.py::TestModels -v

# Test only endpoints
pytest test_main.py::TestEndpoints -v

# Test only async functions
pytest test_main.py::TestAsyncFunctions -v
```

## 📊 Test Results Interpretation

### Success Indicators
- ✅ All tests pass (green checkmarks)
- 📈 High coverage percentage (aim for >90%)
- 🚀 Fast execution time

### Common Issues and Solutions

#### Import Errors
```
Import "pytest" could not be resolved
```
**Solution:** Install test dependencies:
```bash
pip install -r requirements-test.txt
```

#### AI Service Mock Issues
```
AttributeError: 'Mock' object has no attribute 'text'
```
**Solution:** Check that mock responses are properly configured in the test

#### Async Test Failures
```
RuntimeError: Event loop is closed
```
**Solution:** Ensure pytest-asyncio is installed and configured properly

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)
- Automatic test discovery
- Async test support
- Verbose output
- Custom markers for test categorization

### Fixtures (`conftest.py`)
- Mock AI model responses
- Sample test data
- Environment variable mocking
- Reusable test client

## 📈 Test Metrics

The test suite includes:
- **50+ individual test cases**
- **4 test classes** for organized testing
- **100% endpoint coverage**
- **Mock-based testing** for external dependencies
- **Async/await testing** for proper async function validation

## 🛠️ Adding New Tests

### For New Endpoints
1. Add endpoint function to imports in `test_main.py`
2. Create test class or add to `TestEndpoints`
3. Mock AI responses appropriately
4. Test success, failure, and edge cases

### For New Models
1. Add model to imports
2. Create tests in `TestModels` class
3. Test validation, required fields, and data types

### Example Test Pattern
```python
@patch('main.model')
def test_new_endpoint_success(self, mock_model):
    """Test successful response from new endpoint"""
    # Setup mock
    mock_response = Mock()
    mock_response.text = "Expected response"
    mock_model.generate_content.return_value = mock_response
    
    # Make request
    response = client.post("/new-endpoint", json={"key": "value"})
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["expected_field"] == "Expected response"
    
    # Verify AI was called correctly
    mock_model.generate_content.assert_called_once()
```

## 🔍 Debugging Tests

### Enable Debug Output
```bash
pytest test_main.py -v -s --tb=long
```

### Run Single Test
```bash
pytest test_main.py::TestEndpoints::test_enhance_prompt_success -v
```

### Print Debug Information
Add `print()` statements in tests for debugging:
```python
def test_debug_example(self):
    response = client.post("/enhance", json={"prompt": "test"})
    print(f"Response: {response.json()}")  # Debug output
    assert response.status_code == 200
```

## 🎯 Quality Standards

- **Coverage Target:** >90% line coverage
- **Test Speed:** All tests should complete in <30 seconds
- **Reliability:** Tests should pass consistently
- **Maintainability:** Clear test names and documentation

## 📝 Test Naming Convention

- `test_<function_name>_<scenario>()` for specific scenarios
- `test_<endpoint_name>_success()` for happy path
- `test_<endpoint_name>_error()` for error cases
- `test_<model_name>_validation()` for model tests

---

**Happy Testing! 🧪✨**

For questions or issues with the test suite, refer to the main project documentation or create an issue in the repository.
