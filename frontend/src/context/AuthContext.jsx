import { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  // Configure axios with token
  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      localStorage.setItem('token', token);
    } else {
      delete axios.defaults.headers.common['Authorization'];
      localStorage.removeItem('token');
    }
  }, [token]);

  // Load user data if token exists
  useEffect(() => {
    async function loadUser() {
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await axios.get('http://localhost:8000/api/auth/me');
        setCurrentUser(response.data);
      } catch (error) {
        console.error('Error loading user', error);
        // If token is invalid, remove it
        setToken(null);
      } finally {
        setLoading(false);
      }
    }

    loadUser();
  }, [token]);

  // Register a new user
  const register = async (email, username, password) => {
    try {
      const response = await axios.post('http://localhost:8000/api/auth/register', {
        email,
        username,
        password
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Registration failed');
    }
  };

  // Login user
  const login = async (username, password) => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      const response = await axios.post('http://localhost:8000/api/auth/token', 
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      const { access_token } = response.data;
      setToken(access_token);
      
      // Fetch user data
      const userResponse = await axios.get('http://localhost:8000/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${access_token}`
        }
      });
      
      setCurrentUser(userResponse.data);
      return userResponse.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  };

  // Logout user
  const logout = () => {
    setToken(null);
    setCurrentUser(null);
  };

  const value = {
    currentUser,
    token,
    loading,
    register,
    login,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
