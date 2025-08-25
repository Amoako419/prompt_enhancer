import React from 'react';
import { BarChart3, PieChart, TrendingUp, LineChart, AlertCircle } from 'lucide-react';
import '../styles/MCPAnalysisTypes.css';

export const DataPreview = ({ data }) => {
  console.log("DataPreview received data:", data);
  
  if (!data || !data.columns || !data.rows) {
    return (
      <div className="no-data">
        <p>No data available for preview</p>
        <p>Debug: data={JSON.stringify(data)}</p>
      </div>
    );
  }

  return (
    <div className="data-table-container">
      <div className="table-metadata">
        <div className="metadata-item">
          <span className="metadata-label">Rows:</span>
          <span className="metadata-value">{data.rows.length}</span>
        </div>
        <div className="metadata-item">
          <span className="metadata-label">Columns:</span>
          <span className="metadata-value">{data.columns.length}</span>
        </div>
      </div>
      
      <div className="table-scroll-container">
        <table className="data-table">
          <thead>
            <tr>
              {data.columns.map((col, idx) => (
                <th key={idx}>{col.name || `Column ${idx + 1}`}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.slice(0, 10).map((row, rowIdx) => (
              <tr key={rowIdx}>
                {data.columns.map((col, colIdx) => (
                  <td key={colIdx}>
                    {row[col.name] !== null && row[col.name] !== undefined 
                      ? String(row[col.name]) 
                      : 'null'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="table-footer">
        Showing {Math.min(10, data.rows.length)} of {data.rows.length} rows
      </div>
      
      <div className="column-info">
        <h4>Column Information</h4>
        <div className="column-list">
          {data.columns.map((col, idx) => (
            <div key={idx} className="column-item">
              <div className="column-name">{col.name}</div>
              <div className="column-type">{col.type || 'mixed'}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export const StatisticalAnalysis = ({ data }) => {
  console.log("StatisticalAnalysis received data:", data);
  
  // Handle new API format
  if (data && data.statistics && data.numeric_columns) {
    return (
      <div className="statistics-container">
        {data.title && (
          <div className="stats-header">
            <h3>{data.title}</h3>
          </div>
        )}
        
        {data.content && (
          <div className="stats-summary">
            <pre className="formatted-content">{data.content}</pre>
          </div>
        )}
        
        <div className="stats-section">
          <h4>Detailed Statistics Table</h4>
          <div className="table-scroll-container">
            <table className="stats-table">
              <thead>
                <tr>
                  <th>Variable</th>
                  <th>Count</th>
                  <th>Mean</th>
                  <th>Std Dev</th>
                  <th>Min</th>
                  <th>25%</th>
                  <th>Median (50%)</th>
                  <th>75%</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {data.numeric_columns.map((columnName) => {
                  const stats = data.statistics[columnName];
                  if (!stats) return null;
                  
                  return (
                    <tr key={columnName}>
                      <td className="var-name">{columnName}</td>
                      <td>{stats.count || 'N/A'}</td>
                      <td>{typeof stats.mean === 'number' ? stats.mean.toFixed(2) : 'N/A'}</td>
                      <td>{typeof stats.std === 'number' ? stats.std.toFixed(2) : 'N/A'}</td>
                      <td>{typeof stats.min === 'number' ? stats.min.toFixed(2) : 'N/A'}</td>
                      <td>{typeof stats['25%'] === 'number' ? stats['25%'].toFixed(2) : 'N/A'}</td>
                      <td>{typeof stats['50%'] === 'number' ? stats['50%'].toFixed(2) : 'N/A'}</td>
                      <td>{typeof stats['75%'] === 'number' ? stats['75%'].toFixed(2) : 'N/A'}</td>
                      <td>{typeof stats.max === 'number' ? stats.max.toFixed(2) : 'N/A'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="stats-insights">
          <h4>Key Insights</h4>
          <div className="insights-grid">
            {data.numeric_columns.map((columnName) => {
              const stats = data.statistics[columnName];
              if (!stats) return null;
              
              const range = stats.max - stats.min;
              const coefficient_of_variation = stats.std / stats.mean;
              
              return (
                <div key={columnName} className="insight-card">
                  <h5>{columnName}</h5>
                  <div className="insight-items">
                    <div className="insight-item">
                      <span className="insight-label">Range:</span>
                      <span className="insight-value">{range.toFixed(2)}</span>
                    </div>
                    <div className="insight-item">
                      <span className="insight-label">CV:</span>
                      <span className="insight-value">{(coefficient_of_variation * 100).toFixed(1)}%</span>
                    </div>
                    <div className="insight-item">
                      <span className="insight-label">Skewness:</span>
                      <span className="insight-value">
                        {stats.mean > stats['50%'] ? 'Right-skewed' : 
                         stats.mean < stats['50%'] ? 'Left-skewed' : 'Symmetric'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }
  
  // Handle case where we just have title and content (for debugging)
  if (data && (data.title || data.content)) {
    return (
      <div className="statistics-container">
        {data.title && (
          <div className="stats-header">
            <h3>{data.title}</h3>
          </div>
        )}
        
        {data.content && (
          <div className="stats-summary">
            <pre className="formatted-content">{data.content}</pre>
          </div>
        )}
        
        {data.rawData && (
          <div className="debug-section">
            <h4>Debug: Raw API Response</h4>
            <pre className="formatted-content">{JSON.stringify(data.rawData, null, 2)}</pre>
          </div>
        )}
      </div>
    );
  }
  
  // Handle legacy format
  if (data && data.numeric && data.categorical) {
    return (
      <div className="statistics-container">
        <div className="stats-section">
          <h5>Numeric Variables</h5>
          <table className="stats-table">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>25%</th>
                <th>Median</th>
                <th>75%</th>
                <th>Max</th>
              </tr>
            </thead>
            <tbody>
              {data.numeric.map((stat, idx) => (
                <tr key={idx}>
                  <td className="var-name">{stat.variable}</td>
                  <td>{stat.count}</td>
                  <td>{stat.mean.toFixed(2)}</td>
                  <td>{stat.std.toFixed(2)}</td>
                  <td>{stat.min.toFixed(2)}</td>
                  <td>{stat.percentile25.toFixed(2)}</td>
                  <td>{stat.median.toFixed(2)}</td>
                  <td>{stat.percentile75.toFixed(2)}</td>
                  <td>{stat.max.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="stats-section">
          <h5>Categorical Variables</h5>
          <div className="categorical-grid">
            {data.categorical.map((cat, idx) => (
              <div key={idx} className="categorical-item">
                <h6>{cat.variable}</h6>
                <div className="categorical-counts">
                  {cat.counts.slice(0, 5).map((count, countIdx) => (
                    <div key={countIdx} className="category-item">
                      <span className="category-name">{count.category}</span>
                      <span className="category-count">{count.count}</span>
                    </div>
                  ))}
                </div>
                {cat.counts.length > 5 && (
                  <div className="more-categories">
                    +{cat.counts.length - 5} more categories
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="no-data">
      <p>No statistical data available</p>
      <p className="debug-info">Debug: {JSON.stringify(data, null, 2)}</p>
    </div>
  );
};

export const CorrelationAnalysis = ({ data }) => {
  console.log("CorrelationAnalysis received data:", data);
  
  // Handle new API format with title and content
  if (data && data.variables && data.correlations) {
    // Make sure we have both variables and correlations arrays
    if (!Array.isArray(data.variables) || !Array.isArray(data.correlations)) {
      console.error("Invalid correlation data format:", data);
      return <div className="no-data">Invalid correlation data format</div>;
    }

    // Helper function to get color based on correlation value
    const getCorrelationColor = (value) => {
      if (value === 1) return 'rgba(255, 255, 255, 0.9)';
      
      const absValue = Math.abs(value);
      if (value > 0) {
        return `rgba(58, 107, 255, ${absValue.toFixed(2)})`;
      } else {
        return `rgba(255, 99, 132, ${absValue.toFixed(2)})`;
      }
    };

    return (
      <div className="correlation-container">
        {data.title && (
          <div className="stats-header">
            <h3>{data.title}</h3>
          </div>
        )}
        
        {data.content && (
          <div className="stats-summary">
            <pre className="formatted-content">{data.content}</pre>
          </div>
        )}
        
        <div className="stats-section">
          <h4>Interactive Correlation Matrix</h4>
          <div className="table-scroll-container">
            <table className="correlation-table">
              <thead>
                <tr>
                  <th></th>
                  {data.variables.map((variable, idx) => (
                    <th key={idx}>{variable}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.correlations.map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    <td className="var-name">{data.variables[rowIdx]}</td>
                    {row.map((value, colIdx) => (
                      <td 
                        key={colIdx} 
                        style={{ 
                          backgroundColor: getCorrelationColor(value),
                          color: Math.abs(value) > 0.7 ? '#ffffff' : 'inherit'
                        }}
                        className="corr-value"
                        title={`${data.variables[rowIdx]} vs ${data.variables[colIdx]}: ${value.toFixed(3)}`}
                      >
                        {value.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {data.strong_correlations && data.strong_correlations.length > 0 && (
          <div className="stats-section">
            <h4>Strong Correlations</h4>
            <div className="strong-correlations">
              {data.strong_correlations.map((corr, idx) => (
                <div key={idx} className="correlation-item">
                  <span className="correlation-pair">{corr.variables.join(' ↔ ')}</span>
                  <span className="correlation-value">{corr.value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }
  
  // Handle case where we just have title and content (for debugging)
  if (data && (data.title || data.content)) {
    return (
      <div className="correlation-container">
        {data.title && (
          <div className="stats-header">
            <h3>{data.title}</h3>
          </div>
        )}
        
        {data.content && (
          <div className="stats-summary">
            <pre className="formatted-content">{data.content}</pre>
          </div>
        )}
        
        {data.rawData && (
          <div className="debug-section">
            <h4>Debug: Raw API Response</h4>
            <pre className="formatted-content">{JSON.stringify(data.rawData, null, 2)}</pre>
          </div>
        )}
      </div>
    );
  }
  
  return (
    <div className="no-data">
      <p>No correlation data available</p>
      <p className="debug-info">Debug: {JSON.stringify(data, null, 2)}</p>
    </div>
  );
};

export const StatisticalTests = ({ title, content, test_results, tests, rawData }) => {
  console.log("StatisticalTests received data:", { title, content, test_results, tests, rawData });
  
  // Handle new API format with title, content, and test_results
  if (title && content && test_results) {
    return (
      <div className="tests-container">
        <div className="tests-header">
          <h3>{title}</h3>
        </div>
        
        <div className="tests-summary">
          <pre className="formatted-content">{content}</pre>
        </div>
        
        {test_results && test_results.length > 0 && (
          <div className="tests-section">
            <h4>Detailed Test Results</h4>
            <div className="table-scroll-container">
              <table className="tests-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Test Statistic</th>
                    <th>P-Value</th>
                    <th>Sample Size</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {test_results.map((test, idx) => (
                    <tr key={idx}>
                      <td className="test-column">{test.column}</td>
                      <td>{typeof test.statistic === 'number' ? test.statistic.toFixed(4) : test.statistic}</td>
                      <td className={`p-value ${test.p_value <= 0.05 ? 'significant' : 'not-significant'}`}>
                        {typeof test.p_value === 'number' ? 
                          (test.p_value < 0.001 ? test.p_value.toExponential(3) : test.p_value.toFixed(6)) 
                          : test.p_value}
                      </td>
                      <td>{test.sample_size}</td>
                      <td className={`test-result ${test.is_normal ? 'normal' : 'not-normal'}`}>
                        {test.is_normal ? 'Normal' : 'Not Normal'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        <div className="tests-insights">
          <h4>Statistical Interpretation</h4>
          <div className="insight-cards">
            {test_results.map((test, idx) => (
              <div key={idx} className="insight-card">
                <h5>{test.column}</h5>
                <div className="insight-items">
                  <div className="insight-item">
                    <span className="insight-label">Significance:</span>
                    <span className={`insight-value ${test.p_value <= 0.05 ? 'significant' : 'not-significant'}`}>
                      {test.p_value <= 0.05 ? 'Significant' : 'Not Significant'}
                    </span>
                  </div>
                  <div className="insight-item">
                    <span className="insight-label">Interpretation:</span>
                    <span className="insight-value">
                      {test.p_value <= 0.05 ? 'Reject H₀ (not normal)' : 'Fail to reject H₀ (normal)'}
                    </span>
                  </div>
                  <div className="insight-item">
                    <span className="insight-label">Confidence:</span>
                    <span className="insight-value">
                      {test.p_value <= 0.001 ? 'Very High' : 
                       test.p_value <= 0.01 ? 'High' : 
                       test.p_value <= 0.05 ? 'Moderate' : 'Low'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {rawData && (
          <div className="debug-section">
            <details>
              <summary>Debug: Raw API Response</summary>
              <pre className="formatted-content">{JSON.stringify(rawData, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    );
  }
  
  // Handle legacy format
  if (tests && tests.length > 0) {
    return (
      <div className="tests-container">
        {tests.map((test, testIdx) => (
          <div key={testIdx} className="test-item">
            <h5>{test.name}</h5>
            <div className="test-results">
              {Object.entries(test.results).map(([key, value], idx) => (
                <div key={idx} className="test-result-row">
                  <span className="test-key">{key}:</span>
                  <span className="test-value">
                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  }
  
  // No data available
  return (
    <div className="tests-container">
      <div className="no-data">
        <AlertCircle size={48} className="text-gray-400" />
        <p>No statistical tests data available</p>
        <p className="debug-info">Debug: Check console for data structure</p>
      </div>
    </div>
  );
};

export const MLModelResults = ({ title, content, model_data, models, visualizations, rawData }) => {
  console.log("MLModelResults received data:", { title, content, model_data, models, visualizations, rawData });
  
  // Handle new API format with title, content, and model_data
  if (title && content && model_data) {
    return (
      <div className="models-container">
        <div className="ml-header">
          <h3>{title}</h3>
        </div>
        
        <div className="ml-summary">
          <pre className="formatted-content">{content}</pre>
        </div>
        
        <div className="ml-details">
          <h4>Model Details</h4>
          <div className="model-info-grid">
            <div className="info-card">
              <h5>Algorithm</h5>
              <p>{model_data.model_type}</p>
            </div>
            
            {model_data.n_clusters && (
              <div className="info-card">
                <h5>Number of Clusters</h5>
                <p>{model_data.n_clusters}</p>
              </div>
            )}
            
            {model_data.inertia && (
              <div className="info-card">
                <h5>Inertia</h5>
                <p>{model_data.inertia.toFixed(2)}</p>
              </div>
            )}
            
            {model_data.n_iterations && (
              <div className="info-card">
                <h5>Iterations</h5>
                <p>{model_data.n_iterations}</p>
              </div>
            )}
          </div>
          
          {model_data.features_used && (
            <div className="features-section">
              <h5>Features Used</h5>
              <div className="features-list">
                {model_data.features_used.map((feature, idx) => (
                  <span key={idx} className="feature-tag">{feature}</span>
                ))}
              </div>
            </div>
          )}
          
          {model_data.cluster_distribution && (
            <div className="cluster-section">
              <h5>Cluster Distribution</h5>
              <div className="cluster-grid">
                {Object.entries(model_data.cluster_distribution).map(([cluster, count], idx) => (
                  <div key={idx} className="cluster-item">
                    <div className="cluster-name">{cluster}</div>
                    <div className="cluster-count">{count} samples</div>
                    <div className="cluster-percentage">
                      {((count / Object.values(model_data.cluster_distribution).reduce((a, b) => a + b, 0)) * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Display ML Visualizations */}
        {visualizations && visualizations.length > 0 && (
          <div className="ml-visualizations">
            <h4>Machine Learning Visualizations</h4>
            <div className="visualizations-grid">
              {visualizations.map((viz, idx) => (
                <div key={idx} className="visualization-item">
                  <div className="viz-header-item">
                    <div className="viz-icon">
                      {viz.type === 'cluster_scatter' && <TrendingUp size={18} />}
                      {viz.type === 'cluster_distribution' && <BarChart3 size={18} />}
                      {viz.type === 'cluster_features' && <LineChart size={18} />}
                      {viz.type === 'cluster_centers' && <PieChart size={18} />}
                      {!['cluster_scatter', 'cluster_distribution', 'cluster_features', 'cluster_centers'].includes(viz.type) && <BarChart3 size={18} />}
                    </div>
                    <h4>{viz.title}</h4>
                  </div>
                  
                  {viz.image ? (
                    <div className="viz-image-container">
                      <img 
                        src={`data:image/png;base64,${viz.image}`}
                        alt={viz.title}
                        className="viz-image"
                        onError={(e) => {
                          console.error("Failed to load ML image for:", viz.title);
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                      />
                      <div className="viz-fallback" style={{ display: 'none' }}>
                        <div className="fallback-content">
                          <PieChart size={48} className="text-gray-400" />
                          <p>Failed to load visualization</p>
                          <p className="text-sm">Type: {viz.type}</p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="viz-placeholder">
                      <div className="placeholder-content">
                        <PieChart size={48} className="text-gray-400" />
                        <p>No image data available</p>
                        <p className="text-sm">Type: {viz.type}</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {rawData && (
          <div className="debug-section">
            <details>
              <summary>Debug: Raw API Response</summary>
              <pre className="formatted-content">{JSON.stringify(rawData, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    );
  }
  
  // Handle legacy format with models array
  if (models && models.length > 0) {
    // Helper function to get the max absolute value for feature importance scaling
    const getMaxFeatureImportance = (features) => {
      return Math.max(...features.map(f => Math.abs(f.importance)));
    };

    return (
      <div className="models-container">
        {models.map((model, modelIdx) => (
          <div key={modelIdx} className="model-item">
            <h5>{model.name}</h5>
            
            <div className="model-section">
              <h6>Performance Metrics</h6>
              <div className="metrics-grid">
                {Object.entries(model.metrics).map(([key, value], idx) => (
                  <div key={idx} className="metric-item">
                    <span className="metric-name">{key}</span>
                    <span className="metric-value">
                      {typeof value === 'number' ? value.toFixed(3) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {model.featureImportance && model.featureImportance.length > 0 && (
              <div className="model-section">
                <h6>Feature Importance</h6>
                <div className="feature-importance">
                  {(() => {
                    const maxImportance = getMaxFeatureImportance(model.featureImportance);
                    return model.featureImportance.map((feature, fidx) => (
                      <div key={fidx} className="feature-item">
                        <span className="feature-name">{feature.feature}</span>
                        <div className="feature-bar-container">
                          <div 
                            className={`feature-bar ${feature.importance >= 0 ? 'positive' : 'negative'}`}
                            style={{ 
                              width: `${(Math.abs(feature.importance) / maxImportance) * 100}%`,
                              marginLeft: feature.importance < 0 ? 'auto' : '0',
                            }}
                          ></div>
                          <span className="feature-value">{feature.importance.toFixed(3)}</span>
                        </div>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }
  
  // No data available
  return (
    <div className="models-container">
      <div className="no-data">
        <AlertCircle size={48} className="text-gray-400" />
        <p>No model results available</p>
        <p className="debug-info">Debug: Check console for data structure</p>
      </div>
    </div>
  );
};

export const Visualizations = ({ title, content, visualizations }) => {
  console.log("Visualizations component received:", { title, content, visualizations });
  
  if (!visualizations || visualizations.length === 0) {
    return (
      <div className="visualizations-container">
        <div className="no-data">
          <PieChart size={48} className="text-gray-400" />
          <p>No visualization data available</p>
          <p className="debug-info">Debug: Expected visualizations array, got: {JSON.stringify(visualizations)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="visualizations-container">
      {title && (
        <div className="viz-header">
          <h3>{title}</h3>
        </div>
      )}
      
      {content && (
        <div className="viz-summary">
          <pre className="formatted-content">{content}</pre>
        </div>
      )}
      
      <div className="visualizations-grid">
        {visualizations.map((viz, idx) => (
          <div key={idx} className="visualization-item">
            <div className="viz-header-item">
              <div className="viz-icon">
                {viz.type === 'histogram_kde' && <BarChart3 size={18} />}
                {viz.type === 'heatmap' && <TrendingUp size={18} />}
                {viz.type === 'boxplot' && <LineChart size={18} />}
                {viz.type === 'scatter' && <TrendingUp size={18} />}
                {viz.type === 'categorical' && <PieChart size={18} />}
                {!['histogram_kde', 'heatmap', 'boxplot', 'scatter', 'categorical'].includes(viz.type) && <BarChart3 size={18} />}
              </div>
              <h4>{viz.title}</h4>
            </div>
            
            {viz.image ? (
              <div className="viz-image-container">
                <img 
                  src={`data:image/png;base64,${viz.image}`}
                  alt={viz.title}
                  className="viz-image"
                  onError={(e) => {
                    console.error("Failed to load image for:", viz.title);
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
                <div className="viz-fallback" style={{ display: 'none' }}>
                  <div className="fallback-content">
                    <PieChart size={48} className="text-gray-400" />
                    <p>Failed to load visualization</p>
                    <p className="text-sm">Type: {viz.type}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="viz-placeholder">
                <div className="placeholder-content">
                  <PieChart size={48} className="text-gray-400" />
                  <p>No image data available</p>
                  <p className="text-sm">Type: {viz.type}</p>
                </div>
              </div>
            )}
            
            {viz.data && (
              <div className="viz-metadata">
                <p className="text-sm text-gray-600">
                  Additional data available - check console for details
                </p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export const TextAnalysis = ({ data }) => {
  if (!data || !data.insights || data.insights.length === 0) {
    return <div className="no-data">No text analysis data available</div>;
  }

  return (
    <div className="text-analysis-container">
      {data.insights.map((insight, idx) => (
        <div key={idx} className="result-section">
          <h3 className="section-title">{insight.title}</h3>
          <div className="result-content" dangerouslySetInnerHTML={{ __html: insight.content }} />
        </div>
      ))}
    </div>
  );
};
