import { useState } from "react";
import AITools from './components/AITools';
import PromptEnhancer from './components/PromptEnhancer';
import "./App.css";

export default function App() {
  const [currentView, setCurrentView] = useState('tools'); // 'tools' or 'enhancer'

  if (currentView === 'tools') {
    return <AITools onNavigateToEnhancer={() => setCurrentView('enhancer')} />;
  }

  if (currentView === 'enhancer') {
    return <PromptEnhancer onBackToTools={() => setCurrentView('tools')} />;
  }

  return null;
}
