import { useState, useEffect } from 'react';
import { ThemeContext } from './ThemeContext';
import { AuthContext } from './AuthContext';

// Combined Provider component
export const AppProviders = ({ children }) => {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');
  
  // Set theme on body element
  useEffect(() => {
    document.body.classList.remove('light', 'dark');
    document.body.classList.add(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);
  
  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  // Auth context placeholder - will be replaced by the actual AuthProvider
  const authContextValue = {
    currentUser: null,
    token: null,
    loading: true,
    register: () => {},
    login: () => {},
    logout: () => {}
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <AuthContext.Provider value={authContextValue}>
        {children}
      </AuthContext.Provider>
    </ThemeContext.Provider>
  );
};
