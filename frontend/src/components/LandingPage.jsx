import { useState } from 'react';
import { ArrowRight, Sparkles, Database, LineChart, BookOpen, Code, BarChart3, Github } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import Auth from './Auth';
import '../styles/LandingPage.css';

export default function LandingPage({ onEnterApp, setCurrentView }) {
  const [showAuth, setShowAuth] = useState(false);
  const { currentUser } = useAuth();
  
  const handleGetStarted = () => {
    if (currentUser) {
      // If user is already logged in, navigate directly to the app
      onEnterApp();
    } else {
      // Otherwise show the auth component
      setShowAuth(true);
    }
  };

  if (showAuth) {
    return (
      <div className="landing-auth-container">
        <button className="back-to-landing" onClick={() => setShowAuth(false)}>
          Back to Home
        </button>
        <Auth />
      </div>
    );
  }

  return (
    <div className="landing-container">
      <nav className="landing-nav">
        <div className="landing-logo">
          <Sparkles size={24} />
          <span>DataProcAI</span>
        </div>
        <div className="landing-nav-buttons">
          {!currentUser ? (
            <button 
              className="landing-nav-button"
              onClick={() => setShowAuth(true)}
            >
              Log In
            </button>
          ) : (
            <div className="landing-user-buttons">
              <span className="welcome-message">Welcome, {currentUser.username}!</span>
              <button 
                className="landing-nav-button explore-button"
                onClick={handleGetStarted}
              >
                Explore Tools
              </button>
            </div>
          )}
        </div>
      </nav>

      <header className="landing-hero">
        <div className="landing-hero-content">
          <h1>Advanced Data Processing with AI</h1>
          <p>
            An all-in-one platform for data scientists and analysts to process data,
            analyze datasets, generate data pipelines, and more.
          </p>
          <button 
            className="landing-cta-button"
            onClick={handleGetStarted}
          >
            {currentUser ? "Explore Tools" : "Get Started"} <ArrowRight size={16} />
          </button>
        </div>
        <div className="landing-hero-image">
          <div className="hero-image-placeholder">
            {/* This would be replaced with an actual image in production */}
            <LineChart size={120} />
          </div>
        </div>
      </header>

      <section className="landing-features">
        <h2>Key Features</h2>
        <div className="feature-grid">
          <div 
            className={`feature-card ${currentUser ? 'feature-card-clickable' : ''}`}
            onClick={() => {
              if (currentUser) {
                setCurrentView('enhancer');
                onEnterApp();
              }
            }}
          >
            <div className="feature-icon">
              <Sparkles size={32} />
            </div>
            <h3>AI-Powered Processing</h3>
            <p>Optimize your data processing with intelligent AI-driven suggestions and improvements.</p>
            {currentUser && <span className="feature-card-cta">Try it now →</span>}
          </div>
          
          <div 
            className={`feature-card ${currentUser ? 'feature-card-clickable' : ''}`}
            onClick={() => {
              if (currentUser) {
                setCurrentView('sqlConverter');
                onEnterApp();
              }
            }}
          >
            <div className="feature-icon">
              <Code size={32} />
            </div>
            <h3>SQL Converter</h3>
            <p>Convert natural language queries to SQL, T-SQL, and MongoDB queries with ease.</p>
            {currentUser && <span className="feature-card-cta">Try it now →</span>}
          </div>
          
          <div 
            className={`feature-card ${currentUser ? 'feature-card-clickable' : ''}`}
            onClick={() => {
              if (currentUser) {
                setCurrentView('pipelineGenerator');
                onEnterApp();
              }
            }}
          >
            <div className="feature-icon">
              <Database size={32} />
            </div>
            <h3>Data Pipeline Generator</h3>
            <p>Create efficient data pipelines with our intuitive generation tools.</p>
            {currentUser && <span className="feature-card-cta">Try it now →</span>}
          </div>
          
          <div 
            className={`feature-card ${currentUser ? 'feature-card-clickable' : ''}`}
            onClick={() => {
              if (currentUser) {
                setCurrentView('mcpAnalysis');
                onEnterApp();
              }
            }}
          >
            <div className="feature-icon">
              <BarChart3 size={32} />
            </div>
            <h3>MCP Data Analysis</h3>
            <p>Analyze your datasets with comprehensive statistical tools and visualizations.</p>
            {currentUser && <span className="feature-card-cta">Try it now →</span>}
          </div>
          
          <div 
            className={`feature-card ${currentUser ? 'feature-card-clickable' : ''}`}
            onClick={() => {
              if (currentUser) {
                setCurrentView('dataExplorer');
                onEnterApp();
              }
            }}
          >
            <div className="feature-icon">
              <LineChart size={32} />
            </div>
            <h3>Data Explorer</h3>
            <p>Explore your data with interactive visualizations and insights.</p>
            {currentUser && <span className="feature-card-cta">Try it now →</span>}
          </div>
          
          <div 
            className={`feature-card ${currentUser ? 'feature-card-clickable' : ''}`}
            onClick={() => {
              if (currentUser) {
                setCurrentView('skillAssessment');
                onEnterApp();
              }
            }}
          >
            <div className="feature-icon">
              <BookOpen size={32} />
            </div>
            <h3>Skill Assessment</h3>
            <p>Evaluate and improve your data science skills with our assessment tools.</p>
            {currentUser && <span className="feature-card-cta">Try it now →</span>}
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-logo">
            <Sparkles size={20} />
            <span>DataProcAI</span>
          </div>
          <div className="footer-links">
            <a href="https://github.com/Amoako419/prompt_enhancer/blob/main/README.md" target="_blank" rel="noopener noreferrer">
              <Github size={14} /> About
            </a>
            <a href="https://github.com/Amoako419/prompt_enhancer#features" target="_blank" rel="noopener noreferrer">
              <Github size={14} /> Features
            </a>
            <a href="https://github.com/Amoako419/prompt_enhancer/wiki" target="_blank" rel="noopener noreferrer">
              <Github size={14} /> Documentation
            </a>
            <a href="https://github.com/Amoako419/prompt_enhancer/issues" target="_blank" rel="noopener noreferrer">
              <Github size={14} /> Contact
            </a>
          </div>
          <div className="footer-copyright">
            © {new Date().getFullYear()} DataProcAI. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
