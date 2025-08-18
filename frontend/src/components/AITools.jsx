import { MessageSquare, GitBranch, Brain, Database, Users, Workflow, Wand2, BarChart3, Award } from 'lucide-react';
import '../styles/AITools.css';

export default function AITools({ onNavigateToEnhancer, onNavigateToSqlConverter, onNavigateToDataExplorer, onNavigateToSkillAssessment, onNavigateToPipelineGenerator }) {
  const tools = [
    {
      icon: <Wand2 className="tool-icon" />,
      title: 'Prompt Enhancer',
      description: 'Enhance your prompts with AI assistance. Make your prompts more effective and get better results.',
      onClick: onNavigateToEnhancer
    },
    // {
    //   icon: <MessageSquare className="tool-icon" />,
    //   title: 'AmaliAI Chat',
    //   description: 'Supports long prompts up to 150k words and file uploads (~200K tokens), ideal for extensive inputs like codebases or books. Powered by Anthropic.'
    // },
    // {
    //   icon: <GitBranch className="tool-icon" />,
    //   title: 'Code Analysis',
    //   description: 'Analysis all your code repositories from GitHub'
    // },
    {
      icon: <Brain className="tool-icon" />,
      title: 'Skill Assessment',
      description: 'Evaluate your data science and engineering skills with interactive quizzes and hands-on challenges.',
      onClick: onNavigateToSkillAssessment
    },
    // {
    //   icon: <Workflow className="tool-icon" />,
    //   title: 'Deepseek R1',
    //   description: 'Deepseek R1 Distill of Llama 70b'
    // },
    {
      icon: <Database className="tool-icon" />,
      title: 'English to SQL',
      description: 'Convert plain English queries into SQL statements. Perfect for database operations and data analysis.',
      onClick: onNavigateToSqlConverter
    },
    {
      icon: <BarChart3 className="tool-icon" />,
      title: 'Smart Data Exploration',
      description: 'Auto-generate EDA code, statistical analysis, anomaly detection, and data visualization recommendations.',
      onClick: onNavigateToDataExplorer
    },
    {
      icon: <Workflow className="tool-icon" />,
      title: 'Data Pipeline Test Generator',
      description: 'Generate realistic test data for data pipeline validation, ETL testing, and performance analysis.',
      onClick: onNavigateToPipelineGenerator
    }
  ];

  return (
    <div className="tools-container">
      <header className="tools-header">
        <h1>Available AI Tools</h1>
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
