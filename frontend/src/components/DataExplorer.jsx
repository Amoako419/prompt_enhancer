import { useState } from "react";
import axios from "axios";
import { BarChart3, ArrowLeft, Copy, RotateCw, FileText, TrendingUp, AlertTriangle, Eye, Download } from "lucide-react";
import "../App.css";
import "../styles/DataExplorer.css";

export default function DataExplorer({ onBackToTools }) {
  const [description, setDescription] = useState("");
  const [selectedAnalysis, setSelectedAnalysis] = useState("eda");
  const [generatedCode, setGeneratedCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const analysisTypes = [
    {
      id: "eda",
      label: "Exploratory Data Analysis",
      icon: <Eye size={16} />,
      description: "Generate comprehensive EDA code with data overview, distributions, and correlations"
    },
    {
      id: "statistical",
      label: "Statistical Analysis",
      icon: <BarChart3 size={16} />,
      description: "Generate statistical tests, hypothesis testing, and summary statistics"
    },
    {
      id: "anomaly",
      label: "Anomaly Detection",
      icon: <AlertTriangle size={16} />,
      description: "Generate code for outlier detection and anomaly identification"
    },
    {
      id: "timeseries",
      label: "Time Series Analysis",
      icon: <TrendingUp size={16} />,
      description: "Generate time series decomposition, trend analysis, and forecasting code"
    },
    {
      id: "visualization",
      label: "Data Visualization",
      icon: <BarChart3 size={16} />,
      description: "Generate comprehensive visualization code with multiple chart types"
    }
  ];

  const generateAnalysis = async () => {
    if (!description.trim()) return;
    
    setLoading(true);
    try {
      const { data } = await axios.post("http://localhost:8000/data-exploration", {
        description,
        analysis_type: selectedAnalysis
      });
      setGeneratedCode(data.code);
    } catch (err) {
      console.error(err);
      setGeneratedCode("Error generating analysis code. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    if (!generatedCode) return;
    
    try {
      await navigator.clipboard.writeText(generatedCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy: ', err);
    }
  };

  const downloadCode = () => {
    if (!generatedCode) return;
    
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedAnalysis}_analysis.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearAll = () => {
    setDescription("");
    setGeneratedCode("");
    setCopied(false);
  };

  const examplePrompts = [
    {
      type: "eda",
      title: "E-commerce Dataset",
      prompt: "I have a dataset with customer transactions including columns: customer_id, product_name, purchase_date, amount, category, customer_age, location. I want to understand customer behavior patterns."
    },
    {
      type: "statistical",
      title: "A/B Test Analysis",
      prompt: "I have two groups (control and treatment) with conversion rates. I need to perform statistical tests to determine if there's a significant difference between the groups."
    },
    {
      type: "anomaly",
      title: "Network Traffic",
      prompt: "I have network traffic data with timestamps, source_ip, destination_ip, bytes_transferred, and protocol. I need to detect unusual traffic patterns or potential security threats."
    },
    {
      type: "timeseries",
      title: "Sales Forecasting",
      prompt: "I have daily sales data for the past 3 years with columns: date, revenue, units_sold, promotion_flag. I want to analyze trends and forecast future sales."
    },
    {
      type: "visualization",
      title: "Survey Data",
      prompt: "I have survey responses with demographics (age, gender, income), satisfaction scores, and product preferences. I need comprehensive visualizations to present insights."
    }
  ];

  const filteredExamples = examplePrompts.filter(ex => ex.type === selectedAnalysis);

  return (
    <div className="data-explorer-wrapper">
      <header className="chat-header">
        <div className="header-left">
          <button 
            className="back-btn" 
            onClick={onBackToTools}
            title="Back to AI Tools"
          >
            <ArrowLeft size={20} />
            <span className="back-text">Back to Tools</span>
          </button>
          <span>Smart Data Exploration</span>
        </div>
      </header>

      <div className="explorer-container">
        <div className="explorer-content">
          
          {/* Analysis Type Selection */}
          <div className="analysis-selector">
            <h3>Select Analysis Type</h3>
            <div className="analysis-types">
              {analysisTypes.map((type) => (
                <button
                  key={type.id}
                  className={`analysis-type-btn ${selectedAnalysis === type.id ? 'active' : ''}`}
                  onClick={() => setSelectedAnalysis(type.id)}
                >
                  {type.icon}
                  <div>
                    <div className="type-label">{type.label}</div>
                    <div className="type-description">{type.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Input Section */}
          <div className="deinput-section">
            <div className="section-header">
              <FileText className="section-icon" size={20} />
              <h3>Describe Your Dataset and Analysis Goals</h3>
            </div>
            <textarea
              className="description-input"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe your dataset structure, columns, data types, and what insights you're looking for..."
              rows={5}
            />
            <div className="input-actions">
              <button 
                className="generate-btn primary"
                onClick={generateAnalysis}
                disabled={loading || !description.trim()}
              >
                {loading ? (
                  <>
                    <RotateCw size={16} className="spinning" />
                    Generating...
                  </>
                ) : (
                  <>
                    <BarChart3 size={16} />
                    Generate Analysis Code
                  </>
                )}
              </button>
              <button 
                className="clear-btn"
                onClick={clearAll}
                disabled={loading}
              >
                Clear All
              </button>
            </div>
          </div>

          {/* Output Section */}
          {generatedCode && (
            <div className="output-section">
              <div className="section-header">
                <BarChart3 className="section-icon" size={20} />
                <h3>Generated Analysis Code</h3>
                <div className="output-actions">
                  <button 
                    className="action-btn"
                    onClick={copyToClipboard}
                    title="Copy to clipboard"
                  >
                    <Copy size={16} />
                    {copied ? "Copied!" : "Copy"}
                  </button>
                  <button 
                    className="action-btn"
                    onClick={downloadCode}
                    title="Download as Python file"
                  >
                    <Download size={16} />
                    Download
                  </button>
                </div>
              </div>
              <div className="code-output">
                <pre><code>{generatedCode}</code></pre>
              </div>
            </div>
          )}

          {/* Examples Section */}
          <div className="examples-section">
            <h3>Example Prompts for {analysisTypes.find(t => t.id === selectedAnalysis)?.label}</h3>
            <div className="examples-grid">
              {filteredExamples.map((example, index) => (
                <div 
                  key={index}
                  className="example-card"
                  onClick={() => setDescription(example.prompt)}
                >
                  <strong>{example.title}</strong>
                  <span>{example.prompt}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
