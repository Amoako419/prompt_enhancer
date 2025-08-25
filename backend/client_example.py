import requests
import json

# Base URL for API
API_BASE = "http://localhost:8000"

def register_user(email, username, password):
    """Register a new user"""
    response = requests.post(
        f"{API_BASE}/api/auth/register",
        json={"email": email, "username": username, "password": password}
    )
    return response.json()

def login_user(username, password):
    """Login and get access token"""
    response = requests.post(
        f"{API_BASE}/api/auth/token",
        data={"username": username, "password": password}  # Note: form data, not JSON
    )
    return response.json().get("access_token")

def get_user_info(token):
    """Get current user info"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/api/auth/me", headers=headers)
    return response.json()

def save_memory(token, key, value):
    """Save a memory"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_BASE}/api/memory",
        headers=headers,
        json={"key": key, "value": value}
    )
    return response.json()

def get_memory(token, key):
    """Get a memory by key"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/api/memory/{key}", headers=headers)
    return response.json()

def get_all_memories(token):
    """Get all memories"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/api/memory", headers=headers)
    return response.json()

def generate_with_memory(token, prompt, use_history=True, save_history=True):
    """Generate response with memory"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_BASE}/api/ai/generate",
        headers=headers,
        json={
            "prompt": prompt,
            "use_history": use_history,
            "save_history": save_history
        }
    )
    return response.json()

def main():
    """Example usage of the API with memory"""
    # User credentials
    email = "user@example.com"
    username = "testuser"
    password = "password123"
    
    print("1. Registering a user...")
    try:
        user = register_user(email, username, password)
        print(f"User registered: {user}")
    except Exception as e:
        print(f"Registration failed (user may already exist): {e}")
    
    print("\n2. Logging in...")
    token = login_user(username, password)
    print(f"Token received: {token[:10]}...")
    
    print("\n3. Getting user info...")
    user_info = get_user_info(token)
    print(f"User info: {user_info}")
    
    print("\n4. Saving a memory...")
    save_memory(token, "favorite_color", "blue")
    print("Memory saved")
    
    print("\n5. Getting the memory...")
    memory = get_memory(token, "favorite_color")
    print(f"Memory: {memory}")
    
    print("\n6. Generating a response with memory...")
    response = generate_with_memory(token, "Hello, can you remember my favorite color?")
    print(f"Response: {response}")
    
    print("\n7. Generating another response (with conversation history)...")
    response = generate_with_memory(token, "What was my favorite color again?")
    print(f"Response: {response}")
    
    print("\n8. Getting all memories...")
    all_memories = get_all_memories(token)
    print(f"All memories: {json.dumps(all_memories, indent=2)}")


if __name__ == "__main__":
    main()
