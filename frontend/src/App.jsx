import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider } from './context/AuthContext';
import ThemeToggle from './components/ThemeToggle';
import UserProfile from './components/UserProfile';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './components/LandingPage';
import AITools from './components/AITools';
import PromptEnhancerSimple from './components/PromptEnhancerSimple';
import SqlConverter from './components/SqlConverter';
import DataExplorer from './components/DataExplorer';
import SkillAssessment from './components/SkillAssessment';
import DataPipelineGenerator from './components/DataPipelineGenerator';
import MCPDataAnalysis from './components/MCPDataAnalysis';
import "./styles/theme.css";
import "./App.css";
import "./styles/responsive.css";

export default function App() {
  const [showLandingPage, setShowLandingPage] = useState(true);
  const [currentView, setCurrentView] = useState('tools'); // 'tools', 'enhancer', 'sqlConverter', 'dataExplorer', 'skillAssessment', 'pipelineGenerator', or 'mcpAnalysis'

  const renderCurrentView = () => {
    if (currentView === 'tools') {
      return (
        <AITools 
          onNavigateToEnhancer={() => setCurrentView('enhancer')}
          onNavigateToSqlConverter={() => setCurrentView('sqlConverter')}
          onNavigateToDataExplorer={() => setCurrentView('dataExplorer')}
          onNavigateToSkillAssessment={() => setCurrentView('skillAssessment')}
          onNavigateToPipelineGenerator={() => setCurrentView('pipelineGenerator')}
          onNavigateToMCPAnalysis={() => setCurrentView('mcpAnalysis')}
        />
      );
    }

    if (currentView === 'enhancer') {
      return <PromptEnhancerSimple onBackToTools={() => setCurrentView('tools')} />;
    }

    if (currentView === 'sqlConverter') {
      return <SqlConverter onBackToTools={() => setCurrentView('tools')} />;
    }

    if (currentView === 'dataExplorer') {
      return <DataExplorer onBackToTools={() => setCurrentView('tools')} />;
    }

    if (currentView === 'skillAssessment') {
      return <SkillAssessment onBackToTools={() => setCurrentView('tools')} />;
    }

    if (currentView === 'pipelineGenerator') {
      return <DataPipelineGenerator onBack={() => setCurrentView('tools')} />;
    }

    if (currentView === 'mcpAnalysis') {
      return <MCPDataAnalysis onBackToTools={() => setCurrentView('tools')} />;
    }

    return null;
  };

  return (
    <ThemeProvider>
      <AuthProvider>
        {showLandingPage ? (
          <LandingPage 
            onEnterApp={() => setShowLandingPage(false)} 
            setCurrentView={setCurrentView}
          />
        ) : (
          <div className="app">
            {/* Only show header when in the main tools view */}
            {currentView === 'tools' && (
              <header className="app-header">
                <div className="app-header-left">
                  <button 
                    className="back-to-landing-button" 
                    onClick={() => setShowLandingPage(true)}
                  >
                    <ArrowLeft size={16} />
                    Back to Home
                  </button>
                  <h1 className="app-title">AI Tools Suite</h1>
                </div>
                <div className="app-header-right">
                  <ThemeToggle />
                  <UserProfile />
                </div>
              </header>
            )}
            <ProtectedRoute>
              {renderCurrentView()}
            </ProtectedRoute>
          </div>
        )}
      </AuthProvider>
    </ThemeProvider>
  );
}
