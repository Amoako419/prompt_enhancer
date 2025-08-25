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
