import { MessageSquare, GitBranch, Brain, Database, Users, Workflow, Wand2, BarChart3, Award, Server } from 'lucide-react';
import '../styles/AITools.css';

export default function AITools({ onNavigateToEnhancer, onNavigateToSqlConverter, onNavigateToDataExplorer, onNavigateToSkillAssessment, onNavigateToPipelineGenerator, onNavigateToMCPAnalysis }) {
  const tools = [
    {
      icon: <Wand2 className="tool-icon" />,
      title: 'AI Prompt Enhancer',
      description: '🎯 Transform basic prompts into precision-engineered instructions. Boost LLM performance by 300% with smart prompt optimization and conversation history.',
      onClick: onNavigateToEnhancer
    },
    {
      icon: <Brain className="tool-icon" />,
      title: 'Data Skills Assessment',
      description: '📊 Level up your expertise with adaptive challenges. Real-world scenarios in Python, SQL, ML, and statistics. Track progress with detailed analytics.',
      onClick: onNavigateToSkillAssessment
    },
    {
      icon: <Database className="tool-icon" />,
      title: 'Natural Language to SQL',
      description: '💬 Speak data, get SQL. Convert business questions into optimized queries instantly. Support for complex joins, aggregations, and window functions.',
      onClick: onNavigateToSqlConverter
    },
    {
      icon: <BarChart3 className="tool-icon" />,
      title: 'Intelligent EDA Studio',
      description: '🔍 Automated exploratory data analysis that thinks like a senior analyst. Detect patterns, outliers, and generate publication-ready insights in seconds.',
      onClick: onNavigateToDataExplorer
    },
    {
      icon: <Workflow className="tool-icon" />,
      title: 'Pipeline Test Data Factory',
      description: '🏭 Generate realistic test datasets for bulletproof pipelines. Schema-aware, constraint-compliant data that mirrors production complexity.',
      onClick: onNavigateToPipelineGenerator
    },
    {
      icon: <Server className="tool-icon" />,
      title: 'Advanced Analytics Engine',
      description: '🧠 Enterprise-grade statistical analysis powered by MCP. From hypothesis testing to ML modeling – your personal data science co-pilot.',
      onClick: onNavigateToMCPAnalysis
    }
  ];

  return (
    <div className="tools-container">
      <header className="tools-header">
        <h1>Data Professional's AI Arsenal</h1>
        <p className="tools-subtitle">
          Transform your data workflow with intelligent automation. From prompt optimization to advanced analytics – 
          <strong> accelerate insights, eliminate repetitive tasks, and unlock the full potential of your data</strong>.
        </p>
        <div className="tools-badge">
          <span className="badge-icon">⚡</span>
          <span>6 Powerful Tools • Production-Ready • Built for Scale</span>
        </div>
      </header>
      <div className="tools-grid">
        {tools.map((tool, index) => (
          <button 
            key={index} 
            className="tool-card"
            onClick={tool.onClick}
          >
            <div className="tool-icon-wrapper">
              {tool.icon}
            </div>
            <h3 className="tool-title">{tool.title}</h3>
            <p className="tool-description">{tool.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
