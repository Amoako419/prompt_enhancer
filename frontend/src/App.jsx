import { useState } from "react";
import AITools from './components/AITools';
import PromptEnhancerSimple from './components/PromptEnhancerSimple';
import SqlConverter from './components/SqlConverter';
import DataExplorer from './components/DataExplorer';
import SkillAssessment from './components/SkillAssessment';
import DataPipelineGenerator from './components/DataPipelineGenerator';
import "./App.css";

export default function App() {
  const [currentView, setCurrentView] = useState('tools'); // 'tools', 'enhancer', 'sqlConverter', 'dataExplorer', 'skillAssessment', or 'pipelineGenerator'

  if (currentView === 'tools') {
    return (
      <AITools 
        onNavigateToEnhancer={() => setCurrentView('enhancer')}
        onNavigateToSqlConverter={() => setCurrentView('sqlConverter')}
        onNavigateToDataExplorer={() => setCurrentView('dataExplorer')}
        onNavigateToSkillAssessment={() => setCurrentView('skillAssessment')}
        onNavigateToPipelineGenerator={() => setCurrentView('pipelineGenerator')}
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

  return null;
}
