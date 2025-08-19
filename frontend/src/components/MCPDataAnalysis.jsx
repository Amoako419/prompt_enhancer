import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { 
  ArrowLeft, 
  Upload, 
  FileText, 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  Database,
  Download,
  Play,
  Loader2,
  AlertCircle,
  CheckCircle,
  Settings
} from "lucide-react";
import "../styles/MCPDataAnalysis.css";

export default function MCPDataAnalysis({ onBackToTools }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysis, setAnalysis] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mcpStatus, setMcpStatus] = useState("disconnected");
  const fileInputRef = useRef(null);

  // Check MCP status on component mount
  useEffect(() => {
    checkMCPStatus();
  }, []);

  // Available analysis types from the MCP server
  const analysisTypes = [
    {
      id: "load_data",
      label: "Load & Preview Data",
      icon: <Database size={16} />,
      description: "Load and display basic information about your dataset"
    },
    {
      id: "descriptive_stats",
      label: "Descriptive Statistics",
      icon: <BarChart3 size={16} />,
      description: "Generate comprehensive statistical summary of your data"
    },
    {
      id: "correlation_analysis",
      label: "Correlation Analysis",
      icon: <TrendingUp size={16} />,
      description: "Analyze relationships between variables in your dataset"
    },
    {
      id: "visualization",
      label: "Data Visualization",
      icon: <PieChart size={16} />,
      description: "Create various plots and charts for data exploration"
    },
    {
      id: "hypothesis_testing",
      label: "Hypothesis Testing",
      icon: <AlertCircle size={16} />,
      description: "Perform statistical tests (t-test, ANOVA, chi-square)"
    },
    {
      id: "machine_learning",
      label: "Machine Learning",
      icon: <Settings size={16} />,
      description: "Apply ML algorithms for classification, regression, or clustering"
    }
  ];

  // Check MCP server status
  const checkMCPStatus = async () => {
    try {
      setLoading(true);
      const response = await axios.get("http://localhost:8000/mcp/health");
      setMcpStatus("connected");
      setError("");
    } catch (error) {
      setMcpStatus("disconnected");
      setError("MCP server is not running. Please start the backend server first.");
    } finally {
      setLoading(false);
    }
  };

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const allowedTypes = ['.csv', '.xlsx', '.json'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      
      if (allowedTypes.includes(fileExtension)) {
        setSelectedFile(file);
        setError("");
      } else {
        setError("Please select a CSV, Excel, or JSON file.");
        setSelectedFile(null);
      }
    }
  };

  // Execute analysis via MCP server using specific endpoints
  const executeAnalysis = async () => {
    if (!selectedFile) {
      setError("Please upload a file first.");
      return;
    }

    if (!analysis) {
      setError("Please select an analysis type first.");
      return;
    }

    setLoading(true);
    setError("");
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Use the specific endpoint for each analysis type
      const endpointMap = {
        'load_data': '/mcp/load-data',
        'descriptive_stats': '/mcp/descriptive-stats',
        'correlation_analysis': '/mcp/correlation-analysis',
        'visualization': '/mcp/visualization',
        'hypothesis_testing': '/mcp/hypothesis-testing',
        'machine_learning': '/mcp/machine-learning'
      };

      const endpoint = endpointMap[analysis];
      if (!endpoint) {
        throw new Error(`Unknown analysis type: ${analysis}`);
      }

      console.log(`Calling endpoint: http://localhost:8000${endpoint}`); // Debug log

      const response = await axios.post(`http://localhost:8000${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log("Backend Response:", response.data); // Debug log
      
      // Set the complete response data
      setResults(response.data);
      
    } catch (error) {
      console.error("Analysis error:", error);
      setError(error.response?.data?.detail || "Failed to execute analysis. Make sure the backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  // Render analysis results
  const renderResults = () => {
    if (!results) return null;

    console.log("Rendering results:", results); // Debug log

    return (
      <div className="results-section">
        <h3 className="results-title">Analysis Results</h3>
        
        {/* Status Display */}
        {results.status && (
          <div className="result-card">
            <h4>Status</h4>
            <pre className="result-text">{results.status}</pre>
          </div>
        )}

        {/* Main Results */}
        {results.results && results.results.title && (
          <div className="result-card">
            <h4>{results.results.title}</h4>
            <pre className="result-text readable">{results.results.content}</pre>
          </div>
        )}

        {/* Raw data display for debugging */}
        {results.results && results.results.data && (
          <div className="result-card">
            <h4>Technical Data</h4>
            <details>
              <summary>Click to view raw data</summary>
              <pre className="result-text">{JSON.stringify(results.results.data, null, 2)}</pre>
            </details>
          </div>
        )}

        {/* Legacy support for old format */}
        {results.results && !results.results.title && (
          <div className="result-card">
            <h4>Analysis Data</h4>
            <pre className="result-text">{JSON.stringify(results.results, null, 2)}</pre>
          </div>
        )}

        {/* Legacy support for old format */}
        {results.summary && (
          <div className="result-card">
            <h4>Summary</h4>
            <pre className="result-text">{results.summary}</pre>
          </div>
        )}

        {results.statistics && (
          <div className="result-card">
            <h4>Statistics</h4>
            <pre className="result-text">{JSON.stringify(results.statistics, null, 2)}</pre>
          </div>
        )}

        {/* Visualizations */}
        {results.visualizations && results.visualizations.length > 0 && (
          <div className="result-card">
            <h4>Visualizations</h4>
            <div className="visualizations-grid">
              {results.visualizations.map((viz, index) => (
                <div key={index} className="visualization-item">
                  <img 
                    src={`data:image/png;base64,${viz.image}`} 
                    alt={viz.title || `Visualization ${index + 1}`}
                    className="visualization-image"
                  />
                  {viz.title && <p className="viz-title">{viz.title}</p>}
                </div>
              ))}
            </div>
          </div>
        )}

        {results.model_results && (
          <div className="result-card">
            <h4>Model Results</h4>
            <pre className="result-text">{JSON.stringify(results.model_results, null, 2)}</pre>
          </div>
        )}

        {/* Error Display */}
        {results.results && results.results.error && (
          <div className="error-message">
            <AlertCircle size={16} />
            <span>{results.results.error}</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="mcp-data-analysis">
      {/* Header */}
      <div className="mcp-header">
        <div className="header-left">
          <button onClick={onBackToTools} className="back-btn">
            <ArrowLeft size={18} />
            Back to Tools
          </button>
          
          <div className="logo-section">
            <div className="logo-icon">
              <Database size={18} />
            </div>
            <span className="logo-text">MCP Data Analysis</span>
          </div>
        </div>
        
        <div className="header-right">
          <div className={`status-indicator ${mcpStatus}`}>
            <div className="status-dot"></div>
            <span>{mcpStatus === "connected" ? "MCP Connected" : "MCP Disconnected"}</span>
          </div>
          <button onClick={checkMCPStatus} className="status-btn" disabled={loading}>
            {loading ? <Loader2 size={16} className="spin" /> : <Settings size={16} />}
            Check Status
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="mcp-content">
        {/* File Upload Section */}
        <div className="upload-section">
          <h2>Upload Your Data</h2>
          <p className="section-description">
            Upload CSV, Excel, or JSON files to analyze with the MCP Data Analysis server.
          </p>
          
          <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
            <Upload size={32} />
            <p>Click to upload or drag and drop</p>
            <p className="upload-hint">Supports CSV, XLSX, JSON files</p>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv,.xlsx,.json"
              style={{ display: 'none' }}
            />
          </div>
          
          {selectedFile && (
            <div className="selected-file">
              <FileText size={16} />
              <span>{selectedFile.name}</span>
              <span className="file-size">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
            </div>
          )}
        </div>

        {/* Analysis Selection */}
        <div className="analysis-section">
          <h2>Select Analysis Type</h2>
          <div className="analysis-grid">
            {analysisTypes.map((type) => (
              <button
                key={type.id}
                className={`analysis-card ${analysis === type.id ? 'selected' : ''}`}
                onClick={() => setAnalysis(type.id)}
              >
                <div className="analysis-icon">{type.icon}</div>
                <h3>{type.label}</h3>
                <p>{type.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Execute Button */}
        <div className="execute-section">
          <button
            onClick={executeAnalysis}
            disabled={!analysis || loading || mcpStatus !== "connected"}
            className="execute-btn"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Play size={16} />
                Execute Analysis
              </>
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Results */}
        {renderResults()}
      </div>
    </div>
  );
}
