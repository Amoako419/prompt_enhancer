import { MessageSquare, GitBranch, Brain, Database, Users, Workflow, Wand2, BarChart3, Award, Server, Check, ArrowRight, ChevronRight, Sparkles, BookOpen, LineChart } from 'lucide-react';
import '../styles/AITools.css';

export default function AITools({ onNavigateToEnhancer, onNavigateToSqlConverter, onNavigateToDataExplorer, onNavigateToSkillAssessment, onNavigateToPipelineGenerator, onNavigateToMCPAnalysis }) {
  // Tool details with expanded information for sections
  const toolDetails = {
    promptEnhancer: {
      icon: <Wand2 size={40} />,
      title: "Prompt Enhancer",
      emoji: "🎯",
      description: "Transform basic prompts into powerful AI instructions. Generate precise, context-aware prompts that deliver consistent, high-quality results for data analysis tasks.",
      features: [
        "Context-aware prompt generation for data science tasks",
        "Templates for common analytical questions",
        "Structured output format optimization",
        "Parameter tuning suggestions for LLM APIs"
      ],
      useCases: [
        "Extracting insights from complex datasets",
        "Creating visualization-friendly prompt templates",
        "Generating statistical analysis instructions",
        "Optimizing multi-step data processing workflows"
      ],
      onClick: onNavigateToEnhancer,
      bgColor: "linear-gradient(135deg, #0047AB20, #4B008280)",
      accentColor: "#4B0082"
    },
    skillAssessment: {
      icon: <Brain size={40} />,
      title: "Skill Assessment Hub",
      emoji: "🧠",
      description: "Evaluate and benchmark your data science expertise through adaptive assessments. Get personalized learning paths and skill gap analysis with industry-standard metrics.",
      features: [
        "Adaptive testing that adjusts difficulty based on performance",
        "Detailed competency mapping across 50+ data science skills",
        "Personalized learning recommendations",
        "Industry benchmark comparisons"
      ],
      useCases: [
        "Identifying skill gaps for career advancement",
        "Preparing for data science interviews",
        "Creating targeted learning plans",
        "Team skill assessment for project allocation"
      ],
      onClick: onNavigateToSkillAssessment,
      bgColor: "linear-gradient(135deg, #3A1C7120, #6A5ACD80)",
      accentColor: "#6A5ACD"
    },
    sqlConverter: {
      icon: <Database size={40} />,
      title: "Natural Language to SQL",
      emoji: "🔍",
      description: "Convert plain English queries into optimized SQL statements instantly. Perfect for data analysts who need quick database insights without complex syntax.",
      features: [
        "Support for multiple SQL dialects (MySQL, PostgreSQL, MS SQL)",
        "Query optimization suggestions",
        "Schema detection and validation",
        "Export options for various environments"
      ],
      useCases: [
        "Ad-hoc data analysis for non-SQL experts",
        "Rapid prototyping of database queries",
        "Teaching tool for SQL beginners",
        "Quick data retrieval for reports and dashboards"
      ],
      onClick: onNavigateToSqlConverter,
      bgColor: "linear-gradient(135deg, #00334420, #005C8F80)",
      accentColor: "#0066A2"
    },
    dataExplorer: {
      icon: <BarChart3 size={40} />,
      title: "Smart Data Explorer",
      emoji: "📈",
      description: "AI-powered exploratory data analysis with automated insights, statistical summaries, anomaly detection, and publication-ready visualizations in seconds.",
      features: [
        "Automated data cleaning and preprocessing",
        "Statistical significance testing",
        "Interactive visualization generation",
        "Pattern and anomaly detection"
      ],
      useCases: [
        "Quick dataset evaluation before deep analysis",
        "Generating report-ready charts and graphs",
        "Identifying outliers and data quality issues",
        "Time-series forecasting and trend analysis"
      ],
      onClick: onNavigateToDataExplorer,
      bgColor: "linear-gradient(135deg, #00464420, #00758F80)",
      accentColor: "#00758F"
    },
    pipelineGenerator: {
      icon: <Workflow size={40} />,
      title: "Data Pipeline Test Generator",
      emoji: "🛠️",
      description: "Generate realistic, diverse test datasets for pipeline validation, ETL testing, and performance benchmarking. Ensure data quality with smart edge cases.",
      features: [
        "Synthetic data generation with realistic distributions",
        "Edge case and error scenario simulation",
        "Scalable test data volume control",
        "Schema-based generation with constraints"
      ],
      useCases: [
        "ETL pipeline validation and stress testing",
        "Data transformation verification",
        "Performance benchmarking under various load conditions",
        "Testing data integrity rules and constraints"
      ],
      onClick: onNavigateToPipelineGenerator,
      bgColor: "linear-gradient(135deg, #2B4B6F20, #345C8D80)",
      accentColor: "#345C8D"
    },
    mcpAnalysis: {
      icon: <Server size={40} />,
      title: "MCP Data Analysis",
      emoji: "⚙️",
      description: "Advanced statistical analysis powered by Model Context Protocol. Perform correlation analysis, hypothesis testing, and machine learning with enterprise-grade accuracy.",
      features: [
        "Integration with Model Context Protocol frameworks",
        "Advanced statistical modeling and hypothesis testing",
        "Automated machine learning workflows",
        "Explainable AI insights and interpretability"
      ],
      useCases: [
        "Complex correlation and causal analysis",
        "Predictive modeling for business outcomes",
        "Multivariate analysis across large datasets",
        "Feature importance and selection for ML models"
      ],
      onClick: onNavigateToMCPAnalysis,
      bgColor: "linear-gradient(135deg, #00334420, #00667280)",
      accentColor: "#006672"
    }
  };

  return (
    <div className="tools-dashboard">
      <header className="dashboard-header">
        <div className="dashboard-welcome">
          <h1 className="welcome-title">
            <span className="welcome-highlight">Your AI Data Toolkit</span>
          </h1>
          <p className="welcome-subtitle">
            Select a specialized tool to solve your specific data challenges. Each tool is tailored for different aspects of the data workflow.
          </p>
        </div>
      </header>
      
      <div className="dashboard-metrics">
        <div className="metric-card">
          <div className="metric-icon">
            <Sparkles size={24} />
          </div>
          <div className="metric-data">
            <h3>Enhanced Productivity</h3>
            <p>Save up to <span className="highlight">13 hours/week</span> on data tasks</p>
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">
            <LineChart size={24} />
          </div>
          <div className="metric-data">
            <h3>Analysis Accuracy</h3>
            <p>Improve insights by <span className="highlight">87%</span> with AI assistance</p>
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-icon">
            <BookOpen size={24} />
          </div>
          <div className="metric-data">
            <h3>Learning Curve</h3>
            <p>Master new data skills <span className="highlight">3x faster</span></p>
          </div>
        </div>
      </div>

      <div className="tool-sections">
        {/* Prompt Enhancer Section */}
        <section className="tool-section" style={{background: toolDetails.promptEnhancer.bgColor}}>
          <div className="tool-content">
            <div className="tool-header">
              <div className="tool-icon-large" style={{backgroundColor: toolDetails.promptEnhancer.accentColor}}>
                {toolDetails.promptEnhancer.icon}
              </div>
              <div>
                <h2>{toolDetails.promptEnhancer.emoji} {toolDetails.promptEnhancer.title}</h2>
                <p className="tool-tagline">Craft perfect AI instructions for data tasks</p>
              </div>
            </div>
            
            <p className="tool-description">{toolDetails.promptEnhancer.description}</p>
            
            <div className="tool-details-container">
              <div className="tool-features">
                <h3>Key Features</h3>
                <ul>
                  {toolDetails.promptEnhancer.features.map((feature, index) => (
                    <li key={index}><Check size={16} /> {feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="tool-use-cases">
                <h3>Use Cases</h3>
                <ul>
                  {toolDetails.promptEnhancer.useCases.map((useCase, index) => (
                    <li key={index}><ChevronRight size={16} /> {useCase}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="tool-action-button" onClick={toolDetails.promptEnhancer.onClick}>
              Open Prompt Enhancer <ArrowRight size={16} />
            </button>
          </div>
        </section>

        {/* Skill Assessment Section */}
        <section className="tool-section" style={{background: toolDetails.skillAssessment.bgColor}}>
          <div className="tool-content">
            <div className="tool-header">
              <div className="tool-icon-large" style={{backgroundColor: toolDetails.skillAssessment.accentColor}}>
                {toolDetails.skillAssessment.icon}
              </div>
              <div>
                <h2>{toolDetails.skillAssessment.emoji} {toolDetails.skillAssessment.title}</h2>
                <p className="tool-tagline">Benchmark your data science expertise</p>
              </div>
            </div>
            
            <p className="tool-description">{toolDetails.skillAssessment.description}</p>
            
            <div className="tool-details-container">
              <div className="tool-features">
                <h3>Key Features</h3>
                <ul>
                  {toolDetails.skillAssessment.features.map((feature, index) => (
                    <li key={index}><Check size={16} /> {feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="tool-use-cases">
                <h3>Use Cases</h3>
                <ul>
                  {toolDetails.skillAssessment.useCases.map((useCase, index) => (
                    <li key={index}><ChevronRight size={16} /> {useCase}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="tool-action-button" onClick={toolDetails.skillAssessment.onClick}>
              Start Skill Assessment <ArrowRight size={16} />
            </button>
          </div>
        </section>

        {/* SQL Converter Section */}
        <section className="tool-section" style={{background: toolDetails.sqlConverter.bgColor}}>
          <div className="tool-content">
            <div className="tool-header">
              <div className="tool-icon-large" style={{backgroundColor: toolDetails.sqlConverter.accentColor}}>
                {toolDetails.sqlConverter.icon}
              </div>
              <div>
                <h2>{toolDetails.sqlConverter.emoji} {toolDetails.sqlConverter.title}</h2>
                <p className="tool-tagline">English to SQL in seconds</p>
              </div>
            </div>
            
            <p className="tool-description">{toolDetails.sqlConverter.description}</p>
            
            <div className="tool-details-container">
              <div className="tool-features">
                <h3>Key Features</h3>
                <ul>
                  {toolDetails.sqlConverter.features.map((feature, index) => (
                    <li key={index}><Check size={16} /> {feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="tool-use-cases">
                <h3>Use Cases</h3>
                <ul>
                  {toolDetails.sqlConverter.useCases.map((useCase, index) => (
                    <li key={index}><ChevronRight size={16} /> {useCase}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="tool-action-button" onClick={toolDetails.sqlConverter.onClick}>
              Open SQL Converter <ArrowRight size={16} />
            </button>
          </div>
        </section>

        {/* Data Explorer Section */}
        <section className="tool-section" style={{background: toolDetails.dataExplorer.bgColor}}>
          <div className="tool-content">
            <div className="tool-header">
              <div className="tool-icon-large" style={{backgroundColor: toolDetails.dataExplorer.accentColor}}>
                {toolDetails.dataExplorer.icon}
              </div>
              <div>
                <h2>{toolDetails.dataExplorer.emoji} {toolDetails.dataExplorer.title}</h2>
                <p className="tool-tagline">Automated insights from your data</p>
              </div>
            </div>
            
            <p className="tool-description">{toolDetails.dataExplorer.description}</p>
            
            <div className="tool-details-container">
              <div className="tool-features">
                <h3>Key Features</h3>
                <ul>
                  {toolDetails.dataExplorer.features.map((feature, index) => (
                    <li key={index}><Check size={16} /> {feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="tool-use-cases">
                <h3>Use Cases</h3>
                <ul>
                  {toolDetails.dataExplorer.useCases.map((useCase, index) => (
                    <li key={index}><ChevronRight size={16} /> {useCase}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="tool-action-button" onClick={toolDetails.dataExplorer.onClick}>
              Explore Your Data <ArrowRight size={16} />
            </button>
          </div>
        </section>

        {/* Pipeline Generator Section */}
        <section className="tool-section" style={{background: toolDetails.pipelineGenerator.bgColor}}>
          <div className="tool-content">
            <div className="tool-header">
              <div className="tool-icon-large" style={{backgroundColor: toolDetails.pipelineGenerator.accentColor}}>
                {toolDetails.pipelineGenerator.icon}
              </div>
              <div>
                <h2>{toolDetails.pipelineGenerator.emoji} {toolDetails.pipelineGenerator.title}</h2>
                <p className="tool-tagline">Realistic test data for bulletproof pipelines</p>
              </div>
            </div>
            
            <p className="tool-description">{toolDetails.pipelineGenerator.description}</p>
            
            <div className="tool-details-container">
              <div className="tool-features">
                <h3>Key Features</h3>
                <ul>
                  {toolDetails.pipelineGenerator.features.map((feature, index) => (
                    <li key={index}><Check size={16} /> {feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="tool-use-cases">
                <h3>Use Cases</h3>
                <ul>
                  {toolDetails.pipelineGenerator.useCases.map((useCase, index) => (
                    <li key={index}><ChevronRight size={16} /> {useCase}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="tool-action-button" onClick={toolDetails.pipelineGenerator.onClick}>
              Generate Test Data <ArrowRight size={16} />
            </button>
          </div>
        </section>

        {/* MCP Analysis Section */}
        <section className="tool-section" style={{background: toolDetails.mcpAnalysis.bgColor}}>
          <div className="tool-content">
            <div className="tool-header">
              <div className="tool-icon-large" style={{backgroundColor: toolDetails.mcpAnalysis.accentColor}}>
                {toolDetails.mcpAnalysis.icon}
              </div>
              <div>
                <h2>{toolDetails.mcpAnalysis.emoji} {toolDetails.mcpAnalysis.title}</h2>
                <p className="tool-tagline">Enterprise-grade statistical analysis</p>
              </div>
            </div>
            
            <p className="tool-description">{toolDetails.mcpAnalysis.description}</p>
            
            <div className="tool-details-container">
              <div className="tool-features">
                <h3>Key Features</h3>
                <ul>
                  {toolDetails.mcpAnalysis.features.map((feature, index) => (
                    <li key={index}><Check size={16} /> {feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="tool-use-cases">
                <h3>Use Cases</h3>
                <ul>
                  {toolDetails.mcpAnalysis.useCases.map((useCase, index) => (
                    <li key={index}><ChevronRight size={16} /> {useCase}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="tool-action-button" onClick={toolDetails.mcpAnalysis.onClick}>
              Start MCP Analysis <ArrowRight size={16} />
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
