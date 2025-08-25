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
      const tableData = tryParseTableData(results.results.data);
      const metadata = results.results.metadata || {
        filename: selectedFile ? selectedFile.name : "",
        rowCount: "",
        columnCount: ""
      };
      
      // Create the format expected by DataPreview component
      return {
        rows: tableData,
        columns: metadata.columnInfo || (Array.isArray(metadata.columnCount) ? 
          metadata.columnCount.map((col, idx) => ({
            name: col.name || `Column ${idx+1}`,
            type: col.type || 'unknown',
            description: col.description || ''
          })) : [])
      };
    }
    return null;
  };

  // Prepare data for Statistical Analysis
  const prepareStatsAnalysisProps = () => {
    if (results?.results?.statistics) {
      const stats = results.results.statistics;
      return {
        numeric: Object.entries(stats.numerical_stats || {}).map(([variable, values]) => ({
          variable,
          count: values.count,
          mean: values.mean,
          std: values.std,
          min: values.min,
          percentile25: values["25%"],
          median: values["50%"],
          percentile75: values["75%"],
          max: values.max
        })),
        categorical: Object.entries(stats.categorical_counts || {}).map(([variable, counts]) => ({
          variable,
          counts: Object.entries(counts).map(([category, count]) => ({
            category,
            count
          }))
        }))
      };
    }
    return null;
  };

  // Prepare data for Correlation Analysis
  const prepareCorrelationProps = () => {
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
    return null;
  };

  // Prepare data for Statistical Tests
  const prepareTestsProps = () => {
    if (results?.results?.tests) {
      return {
        tests: results.results.tests.map(test => ({
          name: test.name,
          results: test.results || test.result // Handle both field names
        }))
      };
    }
    return null;
  };

  // Prepare data for ML Model Results
  const prepareMLModelProps = () => {
    if (results?.results?.models) {
      return {
        models: results.results.models.map(model => ({
          name: model.name,
          metrics: model.metrics,
          featureImportance: model.feature_importance ? 
            Object.entries(model.feature_importance).map(([feature, importance]) => ({
              feature,
              importance
            })) : []
        }))
      };
    }
    return null;
  };

  // Prepare data for Visualizations
  const prepareVisualizationsProps = () => {
    if (!results?.results) {
      return null;
    }
    
    // Try to handle multiple possible data formats gracefully
    const visualizations = results.results.visualizations || 
                          results.results.charts ||
                          results.results.plots ||
                          results.results.figures;
                          
    if (!visualizations) {
      return null;
    }
    
    // Helper function to detect chart type from title
    const detectChartType = (title) => {
      if (!title) return 'Other';
      
      title = title.toLowerCase();
      
      // Distribution charts
      if (title.includes('bar') || title.includes('column') || 
          title.includes('histogram') || title.includes('distribution') || 
          title.includes('pie') || title.includes('donut') || 
          title.includes('frequency')) {
        return 'Distribution';
      } 
      
      // Relationship charts
      else if (title.includes('scatter') || title.includes('relationship') || 
              title.includes('correlation') || title.includes('heatmap') || 
              title.includes('regression') || title.includes('vs') || 
              title.includes('versus') || title.includes('against')) {
        return 'Relationship';
      } 
      
      // Trend charts
      else if (title.includes('line') || title.includes('trend') || 
              title.includes('time series') || title.includes('temporal') || 
              title.includes('over time') || title.includes('monthly') || 
              title.includes('yearly') || title.includes('quarterly')) {
        return 'Trend';
      }
      
      // Advanced analysis
      else if (title.includes('cluster') || title.includes('segment') || 
              title.includes('basket') || title.includes('network') ||
              title.includes('pca') || title.includes('component') ||
              title.includes('dimension')) {
        return 'Advanced';
      }
      
      return 'Other';
    };
    
    // Function to extract image data from various possible formats
    const getImageData = (chart) => {
      return chart.image || chart.imageData || chart.base64 || 
             chart.plot || chart.data || chart.imageSource;
    };
    
    // If visualizations are already grouped by category
    if (Array.isArray(visualizations) && visualizations[0]?.charts) {
      // Data is already in the expected format with groups
      return {
        visualizations: visualizations.map(group => ({
          title: group.title || "Visualizations",
          charts: group.charts.map(chart => ({
            title: chart.title || "Chart",
            description: chart.description || "Visual representation of data patterns",
            imageData: getImageData(chart)
          }))
        }))
      };
    } 
    // If visualizations are in a flat array, group them
    else if (Array.isArray(visualizations)) {
      // Categorize each chart by analyzing its title
      const groupedCharts = {};
      
      visualizations.forEach(chart => {
        const chartType = detectChartType(chart.title);
        if (!groupedCharts[chartType]) {
          groupedCharts[chartType] = [];
        }
        groupedCharts[chartType].push({
          title: chart.title || "Chart",
          description: chart.description || "Visual representation of data patterns",
          imageData: getImageData(chart)
        });
      });
      
      // Convert grouped charts object to array format expected by component
      const groups = Object.entries(groupedCharts).map(([groupName, charts]) => ({
        title: `${groupName} Analysis`,
        charts
      }));
      
      return { visualizations: groups };
    } 
    // If the visualizations object is itself a single chart
    else if (typeof visualizations === 'object' && visualizations.title) {
      return {
        visualizations: [{
          title: "Analysis Results",
          charts: [{
            title: visualizations.title || "Chart",
            description: visualizations.description || "Visual representation of data patterns",
            imageData: getImageData(visualizations)
          }]
        }]
      };
    }
    
    // For any other data format, return null
    return null;
  };

  // Render analysis results
  const renderResults = () => {
    if (!results) return null;

    console.log("Rendering results:", results); // Debug log
    
    // Determine which component to render based on the analysis type
    const renderAnalysisComponent = () => {
      switch(analysis) {
        case 'load_data':
          const dataPreviewProps = prepareDataPreviewProps();
          return <DataPreview data={dataPreviewProps} />;
          
        case 'descriptive_stats':
          const statsProps = prepareStatsAnalysisProps();
          return <StatisticalAnalysis data={statsProps} />;
          
        case 'correlation_analysis':
          const corrProps = prepareCorrelationProps();
          return <CorrelationAnalysis data={corrProps} />;
          
        case 'visualization':
          const vizProps = prepareVisualizationsProps();
          return <Visualizations data={vizProps} />;
          
        case 'hypothesis_testing':
          const testsProps = prepareTestsProps();
          return <StatisticalTests data={testsProps} />;
          
        case 'machine_learning':
          const mlProps = prepareMLModelProps();
          return <MLModelResults data={mlProps} />;
          
        default:
          // If we don't have a specific component, fall back to data preview
          const defaultProps = prepareDataPreviewProps();
          return <DataPreview data={defaultProps} />;
      }
    };

    return (
      <div className="results-section">
        <h2 className="results-title">Analysis Results</h2>
        
        {/* Status Display */}
        {results.status && (
          <div className="result-status">
            <span className={`status-badge ${results.status}`}>{results.status}</span>
          </div>
        )}

        {/* Main content - Analysis-specific component */}
        <div className="result-card full-width">
          <h4>{results.results?.title || analysisTypes.find(type => type.id === analysis)?.label || "Analysis Results"}</h4>
          {(() => {
            try {
              return renderAnalysisComponent();
            } catch (error) {
              console.error(`Error rendering ${analysis} component:`, error);
              return (
                <div className="error-message">
                  <AlertCircle size={16} />
                  <span>Error displaying results. Check console for details.</span>
                </div>
              );
            }
          })()}
        </div>

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
  // Helper function to prepare data preview props
  const prepareDataPreviewProps = () => {
              status: "success",
              results: {
                title: "Correlation Analysis",
                content: "Correlation matrix generated",
                correlations: {
                  "ROW ID": { "ROW ID": 1.0, "DATE KEY": 0.82, "CUSTOMER ID": 0.12, "QUANTITY": -0.03, "SALES": 0.08, "DISCOUNT": 0.02, "PROFIT": 0.06 },
                  "DATE KEY": { "ROW ID": 0.82, "DATE KEY": 1.0, "CUSTOMER ID": 0.14, "QUANTITY": -0.02, "SALES": 0.07, "DISCOUNT": 0.03, "PROFIT": 0.05 },
                  "CUSTOMER ID": { "ROW ID": 0.12, "DATE KEY": 0.14, "CUSTOMER ID": 1.0, "QUANTITY": 0.04, "SALES": 0.29, "DISCOUNT": -0.05, "PROFIT": 0.32 },
                  "QUANTITY": { "ROW ID": -0.03, "DATE KEY": -0.02, "CUSTOMER ID": 0.04, "QUANTITY": 1.0, "SALES": 0.68, "DISCOUNT": 0.15, "PROFIT": 0.56 },
                  "SALES": { "ROW ID": 0.08, "DATE KEY": 0.07, "CUSTOMER ID": 0.29, "QUANTITY": 0.68, "SALES": 1.0, "DISCOUNT": -0.42, "PROFIT": 0.91 },
                  "DISCOUNT": { "ROW ID": 0.02, "DATE KEY": 0.03, "CUSTOMER ID": -0.05, "QUANTITY": 0.15, "DISCOUNT": 1.0, "PROFIT": -0.53 },
                  "PROFIT": { "ROW ID": 0.06, "DATE KEY": 0.05, "CUSTOMER ID": 0.32, "QUANTITY": 0.56, "SALES": 0.91, "DISCOUNT": -0.53, "PROFIT": 1.0 }
                },
                variables: ["ROW ID", "DATE KEY", "CUSTOMER ID", "QUANTITY", "SALES", "DISCOUNT", "PROFIT"],
                visualizations: [
                  {
                    title: "Correlation Heatmap",
                    image: "base64_encoded_image_placeholder",
                    description: "Heatmap showing correlations between numerical variables"
                  }
                ]
              }
            };
            break;
            
          case 'visualization':
            // Visualization response - restructured for the component
            mockResponse = {
              status: "success",
              results: {
                title: "Data Visualization",
                content: "Generated visualizations based on the dataset",
                visualizations: [
                  {
                    title: "Distribution Analysis",
                    charts: [
                      {
                        title: "Sales by Region Bar Chart",
                        image: "base64_encoded_image_placeholder",
                        description: "Bar chart showing the distribution of sales across different regions"
                      },
                      {
                        title: "Customer Segment Pie Chart",
                        image: "base64_encoded_image_placeholder",
                        description: "Pie chart showing the proportion of customers in each segment"
                      },
                      {
                        title: "Revenue Distribution Histogram",
                        image: "base64_encoded_image_placeholder", 
                        description: "Histogram showing the frequency distribution of revenue values"
                      }
                    ]
                  },
                  {
                    title: "Relationship Analysis",
                    charts: [
                      {
                        title: "Sales vs. Profit Scatter Plot",
                        image: "base64_encoded_image_placeholder",
                        description: "Scatter plot examining the correlation between sales and profit values"
                      },
                      {
                        title: "Price vs. Quantity Scatter Plot",
                        image: "base64_encoded_image_placeholder",
                        description: "Scatter plot showing how price affects quantity purchased"
                      }
                    ]
                  },
                  {
                    title: "Trend Analysis",
                    charts: [
                      {
                        title: "Monthly Sales Line Chart",
                        image: "base64_encoded_image_placeholder",
                        description: "Line chart tracking sales performance over monthly intervals"
                      },
                      {
                        title: "Quarterly Revenue Trend",
                        image: "base64_encoded_image_placeholder",
                        description: "Line chart showing revenue trends across quarters"
                      }
                    ]
                  },
                  {
                    title: "Advanced Analysis",
                    charts: [
                      {
                        title: "Market Basket Analysis Heatmap",
                        image: "base64_encoded_image_placeholder",
                        description: "Heatmap showing product affinities and common purchase patterns"
                      },
                      {
                        title: "Customer Segmentation Cluster Plot",
                        image: "base64_encoded_image_placeholder",
                        description: "Cluster plot showing customer segments based on purchasing behavior"
                      }
                    ]
                  }
                ]
              }
            };
            break;
            
          case 'hypothesis_testing':
            // Hypothesis testing response - restructured for the component
            mockResponse = {
              status: "success",
              results: {
                title: "Hypothesis Testing",
                content: "Completed 3 statistical tests",
                tests: [
                  {
                    name: "ANOVA: Sales across Regions",
                    results: {
                      "F-statistic": 24.67,
                      "p-value": 0.000001,
                      "Degrees of Freedom": [2, 9991],
                      "Conclusion": "Reject null hypothesis. There is a significant difference in sales across regions."
                    }
                  },
                  {
                    name: "T-Test: Profit in AMER vs EMEA",
                    results: {
                      "t-statistic": 12.43,
                      "p-value": 0.000001,
                      "Degrees of Freedom": 8993,
                      "Conclusion": "Reject null hypothesis. There is a significant difference in profit between AMER and EMEA regions."
                    }
                  },
                  {
                    name: "Chi-Square Test: Industry vs Discount",
                    results: {
                      "Chi-square": 198.45,
                      "p-value": 0.000001,
                      "Degrees of Freedom": 16,
                      "Conclusion": "Reject null hypothesis. There is a significant association between industry and discount rate."
                    }
                  }
                ]
              }
            };
            break;
            
          case 'machine_learning':
            // Machine learning response - restructured for the component
            mockResponse = {
              status: "success",
              results: {
                title: "Machine Learning Analysis",
                content: "Applied 3 ML models to predict profit",
                models: [
                  {
                    name: "Linear Regression",
                    metrics: {
                      "R-squared": 0.87,
                      "MSE": 4624.89,
                      "RMSE": 67.97,
                      "MAE": 53.21
                    },
                    featureImportance: [
                      { feature: "SALES", importance: 0.72 },
                      { feature: "DISCOUNT", importance: -0.41 },
                      { feature: "QUANTITY", importance: 0.36 },
                      { feature: "CUSTOMER ID", importance: 0.09 },
                      { feature: "DATE KEY", importance: 0.02 }
                    ]
                  },
                  {
                    name: "Random Forest",
                    metrics: {
                      "R-squared": 0.92,
                      "MSE": 2852.14,
                      "RMSE": 53.40,
                      "MAE": 39.65
                    },
                    featureImportance: [
                      { feature: "SALES", importance: 0.68 },
                      { feature: "DISCOUNT", importance: -0.39 },
                      { feature: "QUANTITY", importance: 0.34 },
                      { feature: "REGION", importance: 0.28 },
                      { feature: "INDUSTRY", importance: 0.21 }
                    ]
                  },
                  {
                    name: "XGBoost",
                    metrics: {
                      "R-squared": 0.94,
                      "MSE": 2139.11,
                      "RMSE": 46.25,
                      "MAE": 34.12
                    },
                    featureImportance: [
                      { feature: "SALES", importance: 0.65 },
                      { feature: "DISCOUNT", importance: -0.38 },
                      { feature: "QUANTITY", importance: 0.32 },
                      { feature: "REGION", importance: 0.25 },
                      { feature: "INDUSTRY", importance: 0.19 },
                      { feature: "PRODUCT", importance: 0.17 }
                    ]
                  }
                ],
                visualizations: [
                  {
                    title: "Model Visualizations",
                    charts: [
                      {
                        title: "Feature Importance Plot",
                        image: "base64_encoded_image_placeholder",
                        description: "Bar chart showing importance of each feature in XGBoost model"
                      },
                      {
                        title: "Actual vs Predicted Values",
                        image: "base64_encoded_image_placeholder",
                        description: "Scatter plot comparing actual profit values with those predicted by the model"
                      }
                    ]
                  }
                ]
              }
            };
            break;
            
          default:
            mockResponse = {
              status: "error",
              error: "Unknown analysis type selected"
            };
        }
        
        console.log(`Mock ${analysis} Response:`, mockResponse);
        
        // Set the mock response data
        setResults(mockResponse);
        setLoading(false);
      }, 1000); // Simulate API delay of 1 second
      
      /* Actual API call implementation for future reference
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
      
      // Format the API response to match our component structure
      const formattedResponse = formatApiResponse(response.data, analysis);
      setResults(formattedResponse);
      */
      
      // Helper function to format API responses for our components
      const formatApiResponse = (apiData, analysisType) => {
        // This would transform the actual API response into the format expected by our components
        // For now, we're using mock data with the correct structure
        return apiData;
      };
      
      return; // Return here to prevent the catch block from running when using the mock data
      
    } catch (error) {
      console.error("Analysis error:", error);
      setError(error.response?.data?.detail || "Failed to execute analysis. Make sure the backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  // No longer needed - we are using the imported DataPreview component

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

  // Render analysis results
  const renderResults = () => {
    if (!results) return null;

    console.log("Rendering results:", results); // Debug log
    
    // Process data for different analysis types
    const getDataForAnalysis = () => {
      if (!results || !results.results) return null;
      return results.results;
    };
    
    // Prepare data for Data Preview
    const prepareDataPreviewProps = () => {
      if (results.results && results.results.data) {
        const tableData = tryParseTableData(results.results.data);
        const metadata = results.results.metadata || {
          filename: selectedFile ? selectedFile.name : "SaaS-Sales.csv",
          rowCount: "9994",
          columnCount: "19"
        };
        
        // Create the format expected by DataPreview component
        return {
          rows: tableData,
          columns: metadata.columnInfo || metadata.columnCount.map((col, idx) => ({
            name: col.name || `Column ${idx+1}`,
            type: col.type || 'unknown',
            description: col.description || ''
          }))
        };
      }
      return null;
    };

    // Prepare data for Statistical Analysis
    const prepareStatsAnalysisProps = () => {
      if (results.results && results.results.statistics) {
        const stats = results.results.statistics;
        return {
          numeric: Object.entries(stats.numerical_stats || {}).map(([variable, values]) => ({
            variable,
            count: values.count,
            mean: values.mean,
            std: values.std,
            min: values.min,
            percentile25: values["25%"],
            median: values["50%"],
            percentile75: values["75%"],
            max: values.max
          })),
          categorical: Object.entries(stats.categorical_counts || {}).map(([variable, counts]) => ({
            variable,
            counts: Object.entries(counts).map(([category, count]) => ({
              category,
              count
            }))
          }))
        };
      }
      return null;
    };

    // Prepare data for Correlation Analysis
    const prepareCorrelationProps = () => {
      if (results.results && results.results.correlations) {
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
      return null;
    };

    // Prepare data for Statistical Tests
    const prepareTestsProps = () => {
      if (results.results && results.results.tests) {
        return {
          tests: results.results.tests.map(test => ({
            name: test.name,
            results: test.results || test.result // Handle both field names
          }))
        };
      }
      return null;
    };

    // Prepare data for ML Model Results
    const prepareMLModelProps = () => {
      if (results.results && results.results.models) {
        return {
          models: results.results.models.map(model => ({
            name: model.name,
            metrics: model.metrics,
            featureImportance: Object.entries(model.feature_importance || {}).map(([feature, importance]) => ({
              feature,
              importance
            }))
          }))
        };
      }
      return null;
    };

    // Prepare data for Visualizations
    const prepareVisualizationsProps = () => {
      if (!results?.results) {
        return null;
      }
      
      // Try to handle multiple possible data formats gracefully
      const visualizations = results.results.visualizations || 
                            results.results.charts ||
                            results.results.plots ||
                            results.results.figures;
                            
      if (!visualizations) {
        return null;
      }
      
      // Helper function to detect chart type from title
      const detectChartType = (title) => {
        if (!title) return 'Other';
        
        title = title.toLowerCase();
        
        // Distribution charts
        if (title.includes('bar') || title.includes('column') || 
            title.includes('histogram') || title.includes('distribution') || 
            title.includes('pie') || title.includes('donut') || 
            title.includes('frequency')) {
          return 'Distribution';
        } 
        
        // Relationship charts
        else if (title.includes('scatter') || title.includes('relationship') || 
                title.includes('correlation') || title.includes('heatmap') || 
                title.includes('regression') || title.includes('vs') || 
                title.includes('versus') || title.includes('against')) {
          return 'Relationship';
        } 
        
        // Trend charts
        else if (title.includes('line') || title.includes('trend') || 
                title.includes('time series') || title.includes('temporal') || 
                title.includes('over time') || title.includes('monthly') || 
                title.includes('yearly') || title.includes('quarterly')) {
          return 'Trend';
        }
        
        // Advanced analysis
        else if (title.includes('cluster') || title.includes('segment') || 
                title.includes('basket') || title.includes('network') ||
                title.includes('pca') || title.includes('component') ||
                title.includes('dimension')) {
          return 'Advanced';
        }
        
        return 'Other';
      };
      
      // Function to extract image data from various possible formats
      const getImageData = (chart) => {
        return chart.image || chart.imageData || chart.base64 || 
               chart.plot || chart.data || chart.imageSource;
      };
      
      // If visualizations are already grouped by category
      if (Array.isArray(visualizations) && visualizations[0]?.charts) {
        // Data is already in the expected format with groups
        return {
          visualizations: visualizations.map(group => ({
            title: group.title || "Visualizations",
            charts: group.charts.map(chart => ({
              title: chart.title || "Chart",
              description: chart.description || "Visual representation of data patterns",
              imageData: getImageData(chart)
            }))
          }))
        };
      } 
      // If visualizations are in a flat array, group them
      else if (Array.isArray(visualizations)) {
        // Categorize each chart by analyzing its title
        const groupedCharts = {};
        
        visualizations.forEach(chart => {
          const chartType = detectChartType(chart.title);
          if (!groupedCharts[chartType]) {
            groupedCharts[chartType] = [];
          }
          groupedCharts[chartType].push({
            title: chart.title || "Chart",
            description: chart.description || "Visual representation of data patterns",
            imageData: getImageData(chart)
          });
        });
        
        // Convert grouped charts object to array format expected by component
        const groups = Object.entries(groupedCharts).map(([groupName, charts]) => ({
          title: `${groupName} Analysis`,
          charts
        }));
        
        return { visualizations: groups };
      } 
      // If the visualizations object is itself a single chart
      else if (typeof visualizations === 'object' && visualizations.title) {
        return {
          visualizations: [{
            title: "Analysis Results",
            charts: [{
              title: visualizations.title || "Chart",
              description: visualizations.description || "Visual representation of data patterns",
              imageData: getImageData(visualizations)
            }]
          }]
        };
      }
      
      // For any other data format, return null
      return null;
    };

    // Determine which component to render based on the analysis type
    const renderAnalysisComponent = () => {
      switch(analysis) {
        case 'load_data':
          const dataPreviewProps = prepareDataPreviewProps();
          return <DataPreview data={dataPreviewProps} />;
          
        case 'descriptive_stats':
          const statsProps = prepareStatsAnalysisProps();
          return <StatisticalAnalysis data={statsProps} />;
          
        case 'correlation_analysis':
          const corrProps = prepareCorrelationProps();
          return <CorrelationAnalysis data={corrProps} />;
          
        case 'visualization':
          const vizProps = prepareVisualizationsProps();
          return <Visualizations data={vizProps} />;
          
        case 'hypothesis_testing':
          const testsProps = prepareTestsProps();
          return <StatisticalTests data={testsProps} />;
          
        case 'machine_learning':
          const mlProps = prepareMLModelProps();
          return <MLModelResults data={mlProps} />;
          
        default:
          // If we don't have a specific component, fall back to data preview
          const defaultProps = prepareDataPreviewProps();
          return <DataPreview data={defaultProps} />;
      }
    };

    return (
      <div className="results-section">
        <h2 className="results-title">Analysis Results</h2>
        
        {/* Status Display */}
        {results.status && (
          <div className="result-status">
            <span className={`status-badge ${results.status}`}>{results.status}</span>
          </div>
        )}

        {/* Main content - Analysis-specific component */}
        <div className="result-card full-width">
          <h4>{results.results?.title || analysisTypes.find(type => type.id === analysis)?.label || "Analysis Results"}</h4>
          {(() => {
            try {
              return renderAnalysisComponent();
            } catch (error) {
              console.error(`Error rendering ${analysis} component:`, error);
              return (
                <div className="error-message">
                  <AlertCircle size={16} />
                  <span>Error displaying results. Check console for details.</span>
                </div>
              );
            }
          })()}
        </div>

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
