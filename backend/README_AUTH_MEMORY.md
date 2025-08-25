# Prompt Enhancer API with Authentication and Memory

This API provides enhanced prompt generation with user authentication and persistent memory.

## Features

- User authentication with JWT tokens
- Session management
- User-specific memory storage
- Integration with Google Gemini AI

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` file to `.env` and fill in your settings:
   ```
   cp .env.example .env
   ```
4. Initialize the database:
   ```
   python init_db.py
   ```
5. Start the server:
   ```
   uvicorn main:app --reload
   ```

## API Usage

### Authentication

Register a new user:
```
POST /api/auth/register
{
  "email": "user@example.com",
  "username": "username",
  "password": "password"
}
```

Get a token:
```
POST /api/auth/token
Form data:
  username: username
  password: password
```

Get current user info:
```
GET /api/auth/me
Authorization: Bearer <token>
```

### Memory

Store a memory:
```
POST /api/memory
Authorization: Bearer <token>
{
  "key": "memory_key",
  "value": "memory_value"
}
```

Get a specific memory:
```
GET /api/memory/memory_key
Authorization: Bearer <token>
```

Get all memories:
```
GET /api/memory
Authorization: Bearer <token>
```

Delete a memory:
```
DELETE /api/memory/memory_key
Authorization: Bearer <token>
```

## Memory-Enhanced AI

Generate responses with conversation history:
```
POST /api/ai/generate
Authorization: Bearer <token>
{
  "prompt": "Your prompt text here",
  "use_history": true,
  "save_history": true,
  "history_limit": 5
}
```

## Example Usage with Memory

```python
import requests
import json

# Login
response = requests.post(
    "http://localhost:8000/api/auth/token",
    data={"username": "user", "password": "password"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Store a memory
memory = {"key": "favorite_color", "value": "blue"}
requests.post("http://localhost:8000/api/memory", json=memory, headers=headers)

# Get a memory
response = requests.get("http://localhost:8000/api/memory/favorite_color", headers=headers)
favorite_color = response.json()["value"]
print(f"Favorite color: {favorite_color}")

# Generate response with conversation history
response = requests.post(
    "http://localhost:8000/api/ai/generate",
    headers=headers,
    json={
        "prompt": "Remember my favorite color is blue",
        "save_history": True
    }
)
print(response.json()["response"])

# Ask about it later (using history)
response = requests.post(
    "http://localhost:8000/api/ai/generate",
    headers=headers,
    json={
        "prompt": "What's my favorite color?",
        "use_history": True
    }
)
print(response.json()["response"])  # Should mention blue
```

For a complete example, see the `client_example.py` file.

## API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
