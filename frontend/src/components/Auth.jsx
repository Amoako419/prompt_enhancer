import { useState } from 'react';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { useAuth } from '../context/AuthContext';
import '../styles/Auth.css';

export default function Auth() {
  const [isLoginView, setIsLoginView] = useState(true);
  const { currentUser, logout, loading } = useAuth();

  const toggleForm = () => {
    setIsLoginView(!isLoginView);
  };

  if (loading) {
    return (
      <div className="auth-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  // If user is already logged in, we'll let the parent component handle redirection
  // The ProtectedRoute will render the dashboard automatically
  if (currentUser) {
    return null;
  }

  return (
    <div className="auth-container">
      {isLoginView ? (
        <LoginForm onToggleForm={toggleForm} />
      ) : (
        <RegisterForm onToggleForm={toggleForm} />
      )}
    </div>
  );
}
