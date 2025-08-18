import { useState } from "react";
import AITools from './components/AITools';
import PromptEnhancer from './components/PromptEnhancer';
import SqlConverter from './components/SqlConverter';
import DataExplorer from './components/DataExplorer';
import SkillAssessment from './components/SkillAssessment';
import "./App.css";

export default function App() {
  const [currentView, setCurrentView] = useState('tools'); // 'tools', 'enhancer', 'sqlConverter', 'dataExplorer', or 'skillAssessment'

  if (currentView === 'tools') {
    return (
      <AITools 
        onNavigateToEnhancer={() => setCurrentView('enhancer')}
        onNavigateToSqlConverter={() => setCurrentView('sqlConverter')}
        onNavigateToDataExplorer={() => setCurrentView('dataExplorer')}
        onNavigateToSkillAssessment={() => setCurrentView('skillAssessment')}
      />
    );
  }

  if (currentView === 'enhancer') {
    return <PromptEnhancer onBackToTools={() => setCurrentView('tools')} />;
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

  return null;
}
