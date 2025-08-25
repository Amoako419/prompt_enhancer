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
      id: "chart_recommendations",
      label: "Smart Chart Suggestions",
      icon: <Settings size={16} />,
      description: "Get intelligent chart recommendations based on your data characteristics"
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
        'chart_recommendations': '/mcp/chart-recommendations',
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

  // Helper function to render data as tables
  const renderDataTable = (data, title = "Data") => {
    if (!data) return null;

    // Handle different data structures
    if (Array.isArray(data)) {
      // Array of objects (like CSV data rows)
      if (data.length > 0 && typeof data[0] === 'object') {
        const columns = Object.keys(data[0]);
        return (
          <div className="data-table-container">
            <h5>{title}</h5>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    {columns.map((col, index) => (
                      <th key={index}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.slice(0, 100).map((row, rowIndex) => ( // Limit to 100 rows for performance
                    <tr key={rowIndex}>
                      {columns.map((col, colIndex) => (
                        <td key={colIndex}>{
                          typeof row[col] === 'object' ? 
                            JSON.stringify(row[col]) : 
                            String(row[col] ?? '')
                        }</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {data.length > 100 && (
                <p className="table-note">Showing first 100 rows of {data.length} total rows</p>
              )}
            </div>
          </div>
        );
      }
      // Array of primitives
      return (
        <div className="data-table-container">
          <h5>{title}</h5>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr><th>Value</th></tr>
              </thead>
              <tbody>
                {data.slice(0, 50).map((item, index) => (
                  <tr key={index}>
                    <td>{String(item)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data.length > 50 && (
              <p className="table-note">Showing first 50 items of {data.length} total items</p>
            )}
          </div>
        </div>
      );
    }

    // Handle object data (like dataset info, statistics)
    if (typeof data === 'object' && data !== null) {
      // Check if it's a preview object with rows
      if (data.preview && Array.isArray(data.preview)) {
        return renderDataTable(data.preview, `${title} (Preview)`);
      }
      
      // Check if it has a columns property (dataset info)
      if (data.columns && typeof data.columns === 'object') {
        const columns = Object.keys(data.columns);
        return (
          <div className="data-table-container">
            <h5>{title} - Column Information</h5>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column Name</th>
                    <th>Data Type</th>
                    <th>Non-Null Count</th>
                    <th>Sample Values</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((colName, index) => {
                    const colInfo = data.columns[colName];
                    return (
                      <tr key={index}>
                        <td><strong>{colName}</strong></td>
                        <td>{colInfo.dtype || 'Unknown'}</td>
                        <td>{colInfo.non_null_count || 'N/A'}</td>
                        <td>{
                          colInfo.sample_values ? 
                            colInfo.sample_values.slice(0, 3).join(', ') + 
                            (colInfo.sample_values.length > 3 ? '...' : '') : 
                            'N/A'
                        }</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            {data.shape && (
              <div className="dataset-summary">
                <p><strong>Dataset Shape:</strong> {data.shape[0]} rows × {data.shape[1]} columns</p>
                {data.missing_values && (
                  <p><strong>Missing Values:</strong> {JSON.stringify(data.missing_values)}</p>
                )}
              </div>
            )}
          </div>
        );
      }

      // Check if it has columns_info property (from MCP server response)
      if (data.columns_info && typeof data.columns_info === 'object') {
        const columns = Object.keys(data.columns_info);
        return (
          <div className="data-table-container">
            <h5>{title} - Column Information</h5>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column Name</th>
                    <th>Data Type</th>
                    <th>Non-Null Count</th>
                    <th>Sample Values</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((colName, index) => {
                    const colInfo = data.columns_info[colName];
                    return (
                      <tr key={index}>
                        <td><strong>{colName}</strong></td>
                        <td>{colInfo.dtype || 'Unknown'}</td>
                        <td>{colInfo.non_null_count || 'N/A'}</td>
                        <td>{
                          colInfo.sample_values ? 
                            colInfo.sample_values.slice(0, 3).join(', ') + 
                            (colInfo.sample_values.length > 3 ? '...' : '') : 
                            'N/A'
                        }</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            {data.shape && (
              <div className="dataset-summary">
                <p><strong>Dataset Shape:</strong> {data.shape[0]} rows × {data.shape[1]} columns</p>
                {data.missing_values && (
                  <p><strong>Missing Values:</strong> {JSON.stringify(data.missing_values)}</p>
                )}
              </div>
            )}
          </div>
        );
      }

      // Check if it's statistics data
      if (data.describe || data.summary || data.statistics) {
        const statsData = data.describe || data.summary || data.statistics;
        if (typeof statsData === 'object') {
          const statKeys = Object.keys(statsData);
          const metrics = statKeys.length > 0 ? Object.keys(statsData[statKeys[0]] || {}) : [];
          
          return (
            <div className="data-table-container">
              <h5>{title} - Statistical Summary</h5>
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {statKeys.map((col, index) => (
                        <th key={index}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((metric, index) => (
                      <tr key={index}>
                        <td><strong>{metric}</strong></td>
                        {statKeys.map((col, colIndex) => (
                          <td key={colIndex}>
                            {typeof statsData[col][metric] === 'number' ? 
                              Number(statsData[col][metric]).toFixed(3) : 
                              String(statsData[col][metric] || 'N/A')
                            }
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        }
      }

      // Generic object renderer as key-value table
      const entries = Object.entries(data);
      if (entries.length > 0) {
        return (
          <div className="data-table-container">
            <h5>{title}</h5>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Property</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(([key, value], index) => (
                    <tr key={index}>
                      <td><strong>{key}</strong></td>
                      <td>{
                        typeof value === 'object' ? 
                          JSON.stringify(value, null, 2) : 
                          String(value)
                      }</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      }
    }

    // Fallback to JSON display
    return (
      <div className="data-table-container">
        <h5>{title}</h5>
        <pre className="result-text">{JSON.stringify(data, null, 2)}</pre>
      </div>
    );
  };

  // Render analysis results
  const renderResults = () => {
    if (!results) return null;

    console.log("Rendering results:", results); // Debug log

    return (
      <div className="results-section">
        <h3 className="results-title">Analysis Results</h3>
        
        {/* Status Display - only show if status is error */}
        {results.status === "error" && (
          <div className="result-card error">
            <h4>Error</h4>
            <pre className="result-text">{results.message || "An unknown error occurred"}</pre>
          </div>
        )}

        {/* Success message */}
        {results.status === "success" && results.message && (
          <div className="result-card success">
            <h4>✅ Success</h4>
            <p className="result-text">{results.message}</p>
          </div>
        )}

        {/* Main Results */}
        {results.results && results.results.title && (
          <div className="result-card">
            <h4>{results.results.title}</h4>
            <div className="result-content readable">{results.results.content}</div>
          </div>
        )}

        {/* Enhanced data display with tables */}
        {results.results && (results.results.data || results.results.preview) && (
          <div className="result-card">
            
            {/* Show preview data as main table if available */}
            {results.results.preview && (
              <>
                {renderDataTable(results.results.preview, "Data Preview (First 5 rows)")}
              </>
            )}
            
            {/* Show dataset summary info if available */}
            {(results.results.filename || results.results.shape || results.results.columns) && (
              <div className="data-table-container">
                <h5>Dataset Summary</h5>
                <div className="table-wrapper">
                  <table className="data-table">
                    <tbody>
                      {results.results.filename && (
                        <tr>
                          <td><strong>Filename</strong></td>
                          <td>{results.results.filename}</td>
                        </tr>
                      )}
                      {results.results.shape && (
                        <tr>
                          <td><strong>Shape</strong></td>
                          <td>{results.results.shape[0]} rows × {results.results.shape[1]} columns</td>
                        </tr>
                      )}
                      {results.results.has_headers !== undefined && (
                        <tr>
                          <td><strong>Headers Detected</strong></td>
                          <td>{results.results.has_headers ? 'Yes' : 'No'}</td>
                        </tr>
                      )}
                      {results.results.columns && (
                        <tr>
                          <td><strong>Columns</strong></td>
                          <td>{Array.isArray(results.results.columns) ? results.results.columns.join(', ') : 'N/A'}</td>
                        </tr>
                      )}
                      {results.results.memory_usage && (
                        <tr>
                          <td><strong>Memory Usage</strong></td>
                          <td>{(results.results.memory_usage / 1024).toFixed(2)} KB</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            
            {/* Show column information if available */}
            {results.results.columns_info && (
              <>
                {renderDataTable({columns_info: results.results.columns_info}, "Column Details")}
              </>
            )}
            
            {/* Show dataset metadata if available */}
            {results.results.data && (
              <>
                {renderDataTable(results.results.data, "Dataset Information")}
              </>
            )}
            
            {/* If neither preview nor data, fall back to any results.data */}
            {!results.results.preview && !results.results.data && results.results && (
              <>
                {renderDataTable(results.results, "Analysis Results")}
              </>
            )}
          </div>
        )}

        {/* Legacy support for old format - now with table rendering */}
        {results.results && !results.results.title && (
          <div className="result-card">
            {renderDataTable(results.results, "Analysis Data")}
          </div>
        )}

        {/* Statistics with table formatting */}
        {results.statistics && (
          <div className="result-card">
            {renderDataTable(results.statistics, "Statistical Analysis")}
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

        {/* Chart Recommendations Display */}
        {results.recommendations && (
          <div className="result-card">
            <h4>📊 Smart Chart Recommendations</h4>
            {results.analysis_summary && (
              <p className="recommendation-summary">{results.analysis_summary}</p>
            )}
            
            {results.dataset_info && (
              <div className="dataset-insights">
                <h5>Dataset Insights</h5>
                <div className="insights-grid">
                  <div className="insight-item">
                    <span className="insight-label">Total Rows:</span>
                    <span className="insight-value">{results.dataset_info.total_rows}</span>
                  </div>
                  <div className="insight-item">
                    <span className="insight-label">Total Columns:</span>
                    <span className="insight-value">{results.dataset_info.total_columns}</span>
                  </div>
                  <div className="insight-item">
                    <span className="insight-label">Numeric Columns:</span>
                    <span className="insight-value">{results.dataset_info.numeric_columns}</span>
                  </div>
                  <div className="insight-item">
                    <span className="insight-label">Categorical Columns:</span>
                    <span className="insight-value">{results.dataset_info.categorical_columns}</span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="recommendations-list">
              {results.recommendations.map((rec, index) => (
                <div key={index} className={`recommendation-card priority-${rec.priority}`}>
                  <div className="recommendation-header">
                    <h6>{rec.chart_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h6>
                    <span className={`priority-badge ${rec.priority}`}>
                      {rec.priority.toUpperCase()}
                    </span>
                  </div>
                  <p className="recommendation-reason">{rec.reason}</p>
                  <p className="recommendation-best-for">
                    <strong>Best for:</strong> {rec.best_for}
                  </p>
                  {rec.suitable_columns && rec.suitable_columns.length > 0 && (
                    <div className="suitable-columns">
                      <strong>Suggested columns:</strong> {rec.suitable_columns.join(', ')}
                    </div>
                  )}
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
