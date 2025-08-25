import { useState, useRef, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import '../styles/UserProfile.css';

export default function UserProfile() {
  const { currentUser, logout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [dropdownRef]);
  
  // Always render the component, even without a current user
  // This ensures the avatar is visible on the homepage

  const handleLogout = () => {
    setDropdownOpen(false);
    logout();
  };

  const firstLetter = currentUser ? currentUser.username.charAt(0).toUpperCase() : "A";

  return (
    <div className="user-profile-container" ref={dropdownRef}>
      <button 
        className="user-avatar" 
        onClick={() => setDropdownOpen(!dropdownOpen)}
        aria-label="User menu"
        title={currentUser ? currentUser.username : "Account"}
      >
        <span className="avatar-letter">{firstLetter}</span>
      </button>
      
      {dropdownOpen && (
        <div className="user-dropdown">
          {currentUser ? (
            <>
              <div className="user-info">
                <span className="username">{currentUser.username}</span>
                <span className="email">{currentUser.email}</span>
              </div>
              <div className="dropdown-divider"></div>
              <button 
                className="logout-button"
                onClick={handleLogout}
              >
                Logout
              </button>
            </>
          ) : (
            <div className="user-info">
              <span className="username">Guest User</span>
              <div className="dropdown-divider"></div>
              <div className="auth-message">Please log in</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
