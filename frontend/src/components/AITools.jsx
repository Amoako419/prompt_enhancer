import { MessageSquare, GitBranch, Brain, Database, Users, Workflow, Wand2, BarChart3, Award, Server } from 'lucide-react';
import '../styles/AITools.css';

export default function AITools({ onNavigateToEnhancer, onNavigateToSqlConverter, onNavigateToDataExplorer, onNavigateToSkillAssessment, onNavigateToPipelineGenerator, onNavigateToMCPAnalysis }) {
  const tools = [
    {
      icon: <Wand2 className="tool-icon" />,
      title: '🎯 Prompt Enhancer',
      description: '💡 Transform basic prompts into powerful AI instructions. Generate precise, context-aware prompts that deliver consistent, high-quality results for data analysis tasks.',
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
      title: '🧠 Skill Assessment Hub',
      description: '📊 Evaluate and benchmark your data science expertise through adaptive assessments. Get personalized learning paths and skill gap analysis with industry-standard metrics.',
      onClick: onNavigateToSkillAssessment
    },
    // {
    //   icon: <Workflow className="tool-icon" />,
    //   title: 'Deepseek R1',
    //   description: 'Deepseek R1 Distill of Llama 70b'
    // },
    {
      icon: <Database className="tool-icon" />,
      title: '🔍 Natural Language to SQL',
      description: '⚡ Convert plain English queries into optimized SQL statements instantly. Perfect for data analysts who need quick database insights without complex syntax.',
      onClick: onNavigateToSqlConverter
    },
    {
      icon: <BarChart3 className="tool-icon" />,
      title: '📈 Smart Data Explorer',
      description: '🚀 AI-powered exploratory data analysis with automated insights, statistical summaries, anomaly detection, and publication-ready visualizations in seconds.',
      onClick: onNavigateToDataExplorer
    },
    {
      icon: <Workflow className="tool-icon" />,
      title: '🛠️ Data Pipeline Test Generator',
      description: '🎲 Generate realistic, diverse test datasets for pipeline validation, ETL testing, and performance benchmarking. Ensure data quality with smart edge cases.',
      onClick: onNavigateToPipelineGenerator
    },
    {
      icon: <Server className="tool-icon" />,
      title: '⚙️ MCP Data Analysis',
      description: '🔬 Advanced statistical analysis powered by Model Context Protocol. Perform correlation analysis, hypothesis testing, and machine learning with enterprise-grade accuracy.',
      onClick: onNavigateToMCPAnalysis
    }
  ];

  return (
    <div className="tools-container">
      <header className="tools-header">
        <div className="hero-section">
          <h1 className="hero-title">
            🚀 Professional AI Tools Suite
            <span className="title-gradient"> for Data Professionals</span>
          </h1>
          <p className="hero-subtitle">
            ⚡ Supercharge your data workflow with cutting-edge AI tools designed for analysts, engineers, and scientists.
            <br />
            <span className="stats-highlight">📈 Boost productivity by 300% | 🎯 Reduce analysis time by 80% | 💡 Unlock insights faster</span>
          </p>
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
