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
  Settings,
  Info,
  FileSpreadsheet
} from "lucide-react";
import "../styles/MCPDataAnalysis.css";
import "../styles/MCPAnalysisTypes.css";
import { 
  DataPreview, 
  StatisticalAnalysis, 
  CorrelationAnalysis,
  StatisticalTests,
  MLModelResults, 
  Visualizations, 
  TextAnalysis 
} from "./MCPAnalysisTypes";

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
    setResults(null);
    
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

      console.log(`Calling endpoint: http://localhost:8000${endpoint}`);

      const response = await axios.post(`http://localhost:8000${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log("Backend Response:", response.data);
      
      if (!response.data) {
        throw new Error("No data received from API");
      }
      
      setResults(response.data);
    } catch (error) {
      console.error("Analysis error:", error);
      setError(error.response?.data?.detail || "Failed to execute analysis. Make sure the backend server is running.");
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to convert data to a suitable format for our components
  const tryParseTableData = (data) => {
    if (Array.isArray(data)) {
      return data;
    } else if (typeof data === 'object' && data !== null) {
      if (data.data && Array.isArray(data.data)) {
        return data.data;
      }
    } else if (typeof data === 'string') {
      try {
        const parsed = JSON.parse(data);
        if (Array.isArray(parsed)) {
          return parsed;
        }
      } catch (e) {
        console.log("Data is not valid JSON");
      }
    }
    console.log("Could not parse data");
    return [];
  };

  // Prepare data for Data Preview
  const prepareDataPreviewProps = () => {
    if (results?.results?.data) {
      const apiData = results.results.data;
      
      // Handle the actual API response format
      if (apiData.preview && apiData.columns) {
        // Convert the preview object format to array of rows
        const preview = apiData.preview;
        const columns = apiData.columns;
        
        // Get the number of rows (assuming all columns have the same number of entries)
        const firstColumn = Object.keys(preview)[0];
        const rowCount = firstColumn ? Object.keys(preview[firstColumn]).length : 0;
        
        // Convert to array of row objects
        const rows = [];
        for (let i = 0; i < rowCount; i++) {
          const row = {};
          columns.forEach(column => {
            row[column] = preview[column] ? preview[column][i.toString()] : null;
          });
          rows.push(row);
        }
        
        // Create column info
        const columnInfo = columns.map(col => ({
          name: col,
          type: 'mixed', // We could infer types if needed
          description: ''
        }));
        
        console.log("Converted rows:", rows);
        console.log("Column info:", columnInfo);
        
        return {
          data: {
            rows: rows,
            columns: columnInfo
          }
        };
      }
      
      // Fallback to the old format if the structure is different
      const tableData = tryParseTableData(results.results.data);
      const metadata = results.results.metadata || {
        filename: selectedFile ? selectedFile.name : "",
        rowCount: "",
        columnCount: ""
      };
      
      return {
        data: {
          rows: tableData,
          columns: metadata.columnInfo || (Array.isArray(metadata.columnCount) ? 
            metadata.columnCount.map((col, idx) => ({
              name: col.name || `Column ${idx+1}`,
              type: col.type || 'unknown',
              description: col.description || ''
            })) : [])
        }
      };
    }
    return null;
  };

  // Prepare data for Statistical Analysis
  const prepareStatsAnalysisProps = () => {
    console.log("prepareStatsAnalysisProps - Full results:", results);
    
    // Handle the exact API format you provided
    if (results?.results) {
      const resultsData = results.results;
      console.log("Results data:", resultsData);
      
      // Check if we have the new format with title, content, and data.statistics
      if (resultsData.title && resultsData.content && resultsData.data?.statistics) {
        console.log("Found new API format with statistics");
        return {
          statistics: resultsData.data.statistics,
          numeric_columns: resultsData.data.numeric_columns,
          title: resultsData.title,
          content: resultsData.content
        };
      }
      
      // Check for older formats
      if (resultsData.data?.statistics) {
        console.log("Found statistics in data");
        return {
          statistics: resultsData.data.statistics,
          numeric_columns: resultsData.data.numeric_columns,
          title: resultsData.title,
          content: resultsData.content
        };
      }
      
      // Even if no structured statistics, pass whatever we have for debugging
      console.log("No structured statistics found, passing raw data for debugging");
      return {
        rawData: resultsData,
        title: resultsData.title,
        content: resultsData.content
      };
    }
    
    console.log("No results data found");
    return null;
  };

  // Prepare data for Correlation Analysis
  const prepareCorrelationProps = () => {
    console.log("prepareCorrelationProps - Full results:", results);
    
    // Handle new API format with correlation_matrix
    if (results?.results?.data?.correlation_matrix) {
      const corrMatrix = results.results.data.correlation_matrix;
      console.log("Found correlation matrix:", corrMatrix);
      
      // Extract variable names from the matrix keys
      const variables = Object.keys(corrMatrix);
      
      // Create the correlation matrix as 2D array
      const correlations = variables.map(rowVar => 
        variables.map(colVar => {
          return corrMatrix[rowVar] && corrMatrix[rowVar][colVar] !== undefined 
            ? corrMatrix[rowVar][colVar] 
            : 0;
        })
      );
      
      console.log("Correlation data prepared:", { variables, correlations });
      
      return {
        variables,
        correlations,
        title: results.results.title,
        content: results.results.content,
        strong_correlations: results.results.data.strong_correlations
      };
    }
    
    // Handle old format
    if (results?.results?.correlations) {
      const corrData = results.results.correlations;
      
      // Use provided variables array if available, otherwise extract from keys
      const variables = results.results.variables || Object.keys(corrData);
      
      // Create the correlation matrix using the variables array for consistent ordering
      const correlations = variables.map(variable => 
        variables.map(v => {
          // Handle potential undefined values - default to 0
          return corrData[variable] && corrData[variable][v] !== undefined 
            ? corrData[variable][v] 
            : 0;
        })
      );
      
      console.log("Correlation props:", { variables, correlations });
      
      return {
        variables,
        correlations
      };
    }
    
    console.log("No correlation data found");
    return null;
  };

  // Prepare data for Statistical Tests
  const prepareTestsProps = () => {
    console.log("Preparing tests props from:", results);
    
    // Handle the new API format
    if (results?.results) {
      const resultsData = results.results;
      
      // Check if we have the new format with title, content, and data
      if (resultsData.title && resultsData.content && resultsData.data) {
        return {
          title: resultsData.title,
          content: resultsData.content,
          test_results: resultsData.data.test_results || [],
          rawData: resultsData.data
        };
      }
      
      // Fallback for old format
      if (resultsData.tests) {
        return {
          tests: resultsData.tests.map(test => ({
            name: test.name,
            results: test.results || test.result // Handle both field names
          }))
        };
      }
    }
    
    console.log("No test data found");
    return null;
  };

  // Prepare data for ML Model Results
  const prepareMLModelProps = () => {
    console.log("Preparing ML props from:", results);
    
    // Handle the new API format
    if (results?.results) {
      const resultsData = results.results;
      
      // Check if we have the new format with title, content, and data
      if (resultsData.title && resultsData.content && resultsData.data) {
        return {
          title: resultsData.title,
          content: resultsData.content,
          model_data: resultsData.data,
          visualizations: results.visualizations || [], // Include visualizations
          rawData: resultsData
        };
      }
      
      // Fallback for old format
      if (resultsData.models) {
        return {
          models: resultsData.models.map(model => ({
            name: model.name,
            metrics: model.metrics,
            featureImportance: model.feature_importance ? 
              Object.entries(model.feature_importance).map(([feature, importance]) => ({
                feature,
                importance
              })) : [],
            predictions: model.predictions ? 
              model.predictions.map((pred, idx) => ({
                id: idx,
                actual: pred.actual,
                predicted: pred.predicted
              })) : []
          }))
        };
      }
    }
    
    console.log("No ML model data found");
    return null;
  };

  // Prepare data for Visualizations
  const prepareVisualizationsProps = () => {
    console.log("Preparing visualization props from:", results);
    
    // Handle the new API format where visualizations are at the top level
    if (results?.visualizations && results.visualizations.length > 0) {
      console.log("Found visualizations:", results.visualizations);
      return {
        title: results.results?.title || "Data Visualizations",
        content: results.results?.content || "",
        visualizations: results.visualizations.map(viz => ({
          type: viz.type,
          title: viz.title,
          image: viz.image, // base64 encoded image
          data: viz.data
        }))
      };
    }
    
    // Fallback for old format
    if (results?.results?.visualizations) {
      return {
        visualizations: results.results.visualizations.map(viz => ({
          type: viz.type,
          title: viz.title,
          data: viz.data,
          layout: viz.layout
        }))
      };
    }
    
    console.log("No visualization data found");
    return null;
  };

  // Prepare data for Text Analysis
  const prepareTextAnalysisProps = () => {
    if (results?.results?.text_analysis) {
      const textData = results.results.text_analysis;
      return {
        wordFrequency: textData.word_frequency || {},
        sentimentAnalysis: textData.sentiment || {},
        namedEntities: textData.named_entities || [],
        topics: textData.topics || []
      };
    }
    return null;
  };

  // Render the appropriate component based on analysis type and results
  const renderResults = () => {
    if (!results) return null;
    
    console.log("Rendering results:", results);
    console.log("Analysis type:", analysis);
    
    // Error handling if no results available
    if (!results.results) {
      console.log("No results.results found");
      return (
        <div className="mcp-no-results">
          <AlertCircle size={36} className="text-warning" />
          <p>No analysis results available.</p>
        </div>
      );
    }

    // Render based on analysis type
    switch(analysis) {
      case 'load_data':
        const dataPreviewProps = prepareDataPreviewProps();
        console.log("DataPreview props:", dataPreviewProps);
        return dataPreviewProps ? 
          <DataPreview {...dataPreviewProps} /> : 
          <div className="mcp-no-results"><p>No preview data available</p></div>;
      
      case 'descriptive_stats':
        const statsProps = prepareStatsAnalysisProps();
        console.log("Stats props:", statsProps);
        console.log("Full results structure:", JSON.stringify(results, null, 2));
        return statsProps ? 
          <StatisticalAnalysis data={statsProps} /> : 
          <div className="mcp-no-results">
            <p>No statistical data available</p>
            <p>Debug: Check console for full data structure</p>
          </div>;
      
      case 'correlation_analysis':
        const corrProps = prepareCorrelationProps();
        console.log("Correlation props:", corrProps);
        console.log("Full results structure for correlation:", JSON.stringify(results, null, 2));
        return corrProps ? 
          <CorrelationAnalysis data={corrProps} /> : 
          <div className="mcp-no-results">
            <p>No correlation data available</p>
            <p>Debug: Check console for full data structure</p>
          </div>;
      
      case 'hypothesis_testing':
        const testsProps = prepareTestsProps();
        console.log("Tests props prepared:", testsProps);
        return testsProps ? 
          <StatisticalTests 
            title={testsProps.title}
            content={testsProps.content}
            test_results={testsProps.test_results}
            tests={testsProps.tests}
            rawData={testsProps.rawData}
          /> : 
          <div className="mcp-no-results">
            <p>No test results available</p>
            <p>Debug: Check console for API response structure</p>
          </div>;
      
      case 'machine_learning':
        const mlProps = prepareMLModelProps();
        console.log("ML props prepared:", mlProps);
        return mlProps ? 
          <MLModelResults 
            title={mlProps.title}
            content={mlProps.content}
            model_data={mlProps.model_data}
            models={mlProps.models}
            visualizations={mlProps.visualizations}
            rawData={mlProps.rawData}
          /> : 
          <div className="mcp-no-results">
            <p>No model results available</p>
            <p>Debug: Check console for API response structure</p>
          </div>;
      
      case 'visualization':
        const vizProps = prepareVisualizationsProps();
        console.log("Visualization props prepared:", vizProps);
        return vizProps ? 
          <Visualizations 
            title={vizProps.title}
            content={vizProps.content}
            visualizations={vizProps.visualizations}
          /> : 
          <div className="mcp-no-results">
            <p>No visualization data available</p>
            <p>Debug: Check console for API response structure</p>
          </div>;
      
      default:
        return (
          <div className="mcp-no-results">
            <AlertCircle size={36} className="text-warning" />
            <p>Unknown analysis type: {analysis}</p>
          </div>
        );
    }
  };

  // Download results as JSON
  const downloadResults = () => {
    if (!results) return;
    
    const jsonString = JSON.stringify(results, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `${analysis}-results.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mcp-data-analysis-container">
      {/* Header */}
      <div className="mcp-data-analysis-header">
        <button 
          className="mcp-back-button" 
          onClick={onBackToTools}
        >
          <ArrowLeft size={16} />
          <span>Back to Tools</span>
        </button>
        <h2>MCP Data Analysis</h2>
      </div>

      {/* Status indicator */}
      <div className="mcp-status-bar">
        <div className={`mcp-status-indicator ${mcpStatus}`}>
          {mcpStatus === "connected" ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
          <span>MCP Server: {mcpStatus}</span>
        </div>
        
        {mcpStatus === "disconnected" && (
          <button className="mcp-retry-button" onClick={checkMCPStatus}>
            Retry Connection
          </button>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="mcp-error-message">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Analysis selection */}
      <div className="mcp-analysis-container">
        <div className="mcp-sidebar">
          <div className="mcp-upload-section">
            <h3>Upload Data</h3>
            <div className="mcp-upload-zone" onClick={() => fileInputRef.current.click()}>
              <Upload size={24} />
              <p>{selectedFile ? selectedFile.name : "Click to upload file"}</p>
              <span className="mcp-upload-hint">CSV, Excel or JSON</span>
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
                accept=".csv,.xlsx,.xls,.json"
              />
            </div>
            {selectedFile && (
              <div className="mcp-file-info">
                <FileText size={14} />
                <span>{selectedFile.name}</span>
                <span className="mcp-file-size">{(selectedFile.size / 1024).toFixed(1)} KB</span>
              </div>
            )}
          </div>

          <div className="mcp-analysis-selection">
            <h3>Select Analysis</h3>
            <div className="mcp-analysis-options">
              {analysisTypes.map((type) => (
                <button 
                  key={type.id} 
                  className={`mcp-analysis-option ${analysis === type.id ? 'selected' : ''}`}
                  onClick={() => setAnalysis(type.id)}
                  disabled={mcpStatus === "disconnected" || !selectedFile}
                >
                  <div className="mcp-analysis-option-icon">
                    {type.icon}
                  </div>
                  <div className="mcp-analysis-option-text">
                    <span>{type.label}</span>
                    <span className="mcp-analysis-option-description">{type.description}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <button 
            className="mcp-execute-button"
            onClick={executeAnalysis}
            disabled={!selectedFile || !analysis || loading || mcpStatus === "disconnected"}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="mcp-spinner" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>Execute Analysis</span>
              </>
            )}
          </button>
        </div>

        <div className="mcp-results-section">
          {loading ? (
            <div className="mcp-loading">
              <Loader2 size={36} className="mcp-spinner" />
              <p>Analyzing data...</p>
            </div>
          ) : results ? (
            <div className="mcp-results-container">
              <div className="mcp-results-header">
                <h3>Analysis Results</h3>
                <button className="mcp-download-button" onClick={downloadResults}>
                  <Download size={14} />
                  <span>Download</span>
                </button>
              </div>
              <div className="mcp-results-content">
                {renderResults()}
              </div>
            </div>
          ) : (
            <div className="mcp-placeholder">
              <Info size={36} className="text-muted" />
              <h3>Ready for Analysis</h3>
              <p>Upload a file and select an analysis type to get started</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
