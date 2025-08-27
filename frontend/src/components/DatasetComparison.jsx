import React, { useState } from 'react';
import { 
  ArrowLeft,
  ArrowRightLeft, 
  FileText,
  ChevronDown,
  ChevronUp,
  Loader2
} from 'lucide-react';
import '../styles/DatasetComparison.css';

/**
 * Dataset Comparison Component
 * Displays comparative analysis of two datasets side by side
 */
const DatasetComparison = ({ 
  primaryDataset, 
  secondaryDataset,
  analysisResults,
  analysisType,
  loading
}) => {
  const [showDifferences, setShowDifferences] = useState(true);
  
  if (loading) {
    return (
      <div className="comparison-loading">
        <Loader2 size={30} className="spinning" />
        <p>Comparing datasets...</p>
      </div>
    );
  }

  if (!primaryDataset || !secondaryDataset) {
    return (
      <div className="comparison-placeholder">
        <ArrowRightLeft size={40} />
        <p>Please upload both datasets to see comparison</p>
      </div>
    );
  }

  // Render basic metadata comparison
  const renderMetadataComparison = () => {
    return (
      <div className="metadata-comparison">
        <h3>Dataset Comparison Overview</h3>
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>{primaryDataset.name || "Dataset A"}</th>
              <th>{secondaryDataset.name || "Dataset B"}</th>
              <th>Difference</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Rows</td>
              <td>{primaryDataset.rowCount || "N/A"}</td>
              <td>{secondaryDataset.rowCount || "N/A"}</td>
              <td>
                {primaryDataset.rowCount && secondaryDataset.rowCount ? (
                  <span className={
                    primaryDataset.rowCount > secondaryDataset.rowCount ? "positive-diff" :
                    primaryDataset.rowCount < secondaryDataset.rowCount ? "negative-diff" : ""
                  }>
                    {Math.abs(primaryDataset.rowCount - secondaryDataset.rowCount)}
                    {" "}
                    ({Math.round(Math.abs(primaryDataset.rowCount - secondaryDataset.rowCount) / 
                      primaryDataset.rowCount * 100)}%)
                  </span>
                ) : "N/A"}
              </td>
            </tr>
            <tr>
              <td>Columns</td>
              <td>{primaryDataset.columnCount || "N/A"}</td>
              <td>{secondaryDataset.columnCount || "N/A"}</td>
              <td>
                {primaryDataset.columnCount && secondaryDataset.columnCount ? (
                  <span className={
                    primaryDataset.columnCount > secondaryDataset.columnCount ? "positive-diff" :
                    primaryDataset.columnCount < secondaryDataset.columnCount ? "negative-diff" : ""
                  }>
                    {Math.abs(primaryDataset.columnCount - secondaryDataset.columnCount)}
                  </span>
                ) : "N/A"}
              </td>
            </tr>
            <tr>
              <td>File Size</td>
              <td>{primaryDataset.fileSize || "N/A"}</td>
              <td>{secondaryDataset.fileSize || "N/A"}</td>
              <td>-</td>
            </tr>
          </tbody>
        </table>
      </div>
    );
  };
  
  // Render column comparison
  const renderColumnComparison = () => {
    const primaryColumns = primaryDataset.columns || [];
    const secondaryColumns = secondaryDataset.columns || [];
    
    // Find common and unique columns
    const commonColumns = primaryColumns.filter(col => 
      secondaryColumns.some(scol => scol.name === col.name)
    );
    
    const uniqueToPrimary = primaryColumns.filter(col => 
      !secondaryColumns.some(scol => scol.name === col.name)
    );
    
    const uniqueToSecondary = secondaryColumns.filter(col => 
      !primaryColumns.some(pcol => pcol.name === col.name)
    );
    
    return (
      <div className="column-comparison">
        <div className="comparison-section-header">
          <h3>Column Comparison</h3>
          <button 
            className="toggle-button" 
            onClick={() => setShowDifferences(!showDifferences)}
            title={showDifferences ? "Show all columns" : "Show only differences"}
          >
            {showDifferences ? "Show All" : "Show Differences"}
          </button>
        </div>
        
        {/* Common columns */}
        {(commonColumns.length > 0) && (
          <div className="column-section">
            <h4>Common Columns ({commonColumns.length})</h4>
            <div className="column-grid">
              {commonColumns.map((col, idx) => (
                <div key={idx} className="column-card common-column">
                  <span className="column-name">{col.name}</span>
                  <span className="column-type">Type: {col.type || "unknown"}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Only show differences if requested or if there are no common columns */}
        {(showDifferences || commonColumns.length === 0) && (
          <>
            {/* Unique to primary */}
            {uniqueToPrimary.length > 0 && (
              <div className="column-section">
                <h4>Unique to {primaryDataset.name || "Dataset A"} ({uniqueToPrimary.length})</h4>
                <div className="column-grid">
                  {uniqueToPrimary.map((col, idx) => (
                    <div key={idx} className="column-card unique-to-primary">
                      <span className="column-name">{col.name}</span>
                      <span className="column-type">Type: {col.type || "unknown"}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Unique to secondary */}
            {uniqueToSecondary.length > 0 && (
              <div className="column-section">
                <h4>Unique to {secondaryDataset.name || "Dataset B"} ({uniqueToSecondary.length})</h4>
                <div className="column-grid">
                  {uniqueToSecondary.map((col, idx) => (
                    <div key={idx} className="column-card unique-to-secondary">
                      <span className="column-name">{col.name}</span>
                      <span className="column-type">Type: {col.type || "unknown"}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    );
  };
  
  // Analysis-specific comparisons
  const renderAnalysisComparison = () => {
    if (!analysisResults) return null;
    
    switch (analysisType) {
      case 'descriptive_stats':
        return renderStatsComparison();
      case 'correlation_analysis':
        return renderCorrelationComparison();
      default:
        return null;
    }
  };
  
  // Statistical comparison
  const renderStatsComparison = () => {
    // This would be populated with actual comparison data from the backend
    return (
      <div className="stats-comparison">
        <h3>Statistical Comparison</h3>
        <p className="placeholder-message">
          Statistical comparison visualization would appear here based on backend analysis
        </p>
        <div className="comparison-placeholder-viz">
          <div className="viz-placeholder">Statistical Comparison Chart</div>
        </div>
      </div>
    );
  };
  
  // Correlation comparison
  const renderCorrelationComparison = () => {
    return (
      <div className="correlation-comparison">
        <h3>Correlation Matrix Comparison</h3>
        <p className="placeholder-message">
          Correlation comparison visualization would appear here based on backend analysis
        </p>
        <div className="comparison-placeholder-viz">
          <div className="viz-placeholder">Correlation Diff Heatmap</div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="dataset-comparison-container">
      <div className="comparison-header">
        <h2>
          <ArrowRightLeft size={20} />
          Comparing Datasets
        </h2>
        <div className="dataset-labels">
          <span className="primary-label">{primaryDataset.name || "Dataset A"}</span>
          <span className="vs">vs</span>
          <span className="secondary-label">{secondaryDataset.name || "Dataset B"}</span>
        </div>
      </div>
      
      <div className="comparison-content">
        {renderMetadataComparison()}
        {renderColumnComparison()}
        {renderAnalysisComparison()}
      </div>
    </div>
  );
};

export default DatasetComparison;
