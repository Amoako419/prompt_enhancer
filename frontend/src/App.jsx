import { useState } from "react";
import AITools from './components/AITools';
import PromptEnhancer from './components/PromptEnhancer';
import SqlConverter from './components/SqlConverter';
import DataExplorer from './components/DataExplorer';
import "./App.css";

export default function App() {
  const [currentView, setCurrentView] = useState('tools'); // 'tools', 'enhancer', 'sqlConverter', or 'dataExplorer'

  if (currentView === 'tools') {
    return (
      <AITools 
        onNavigateToEnhancer={() => setCurrentView('enhancer')}
        onNavigateToSqlConverter={() => setCurrentView('sqlConverter')}
        onNavigateToDataExplorer={() => setCurrentView('dataExplorer')}
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

  return null;
}
