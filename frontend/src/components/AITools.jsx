import { MessageSquare, GitBranch, Brain, HelpCircle, Users, Workflow, Wand2 } from 'lucide-react';
import '../styles/AITools.css';

export default function AITools({ onNavigateToEnhancer }) {
  const tools = [
    {
      icon: <Wand2 className="tool-icon" />,
      title: 'Prompt Enhancer',
      description: 'Enhance your prompts with AI assistance. Make your prompts more effective and get better results.',
      onClick: onNavigateToEnhancer
    },
    {
      icon: <MessageSquare className="tool-icon" />,
      title: 'AmaliAI Chat',
      description: 'Supports long prompts up to 150k words and file uploads (~200K tokens), ideal for extensive inputs like codebases or books. Powered by Anthropic.'
    },
    {
      icon: <GitBranch className="tool-icon" />,
      title: 'Code Analysis',
      description: 'Analysis all your code repositories from GitHub'
    },
    {
      icon: <Brain className="tool-icon" />,
      title: 'Skill Assessment',
      description: 'Evaluate your coding skills with interactive quizzes and challenges.'
    },
    {
      icon: <Workflow className="tool-icon" />,
      title: 'Deepseek R1',
      description: 'Deepseek R1 Distill of Llama 70b'
    },
    {
      icon: <Users className="tool-icon" />,
      title: 'AmaliAI HR',
      description: 'Can answer HR-related queries at AmaliTech, covering workplace policies and benefits. Powered by OpenAI.'
    },
    {
      icon: <HelpCircle className="tool-icon" />,
      title: 'Evaluation Tool',
      description: "An evaluation tool that is used to evaluate a model's performances. See how your model really performs."
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
