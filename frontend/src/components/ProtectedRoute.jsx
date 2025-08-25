import { useAuth } from '../context/AuthContext';
import Auth from './Auth';

export default function ProtectedRoute({ children }) {
  const { currentUser, loading } = useAuth();

  if (loading) {
    return (
      <div className="auth-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  // If no user is logged in, show the authentication component
  if (!currentUser) {
    return <Auth />;
  }

  // If user is logged in, render the children components
  return children;
}
