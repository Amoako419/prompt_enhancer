import React, { useState } from 'react';
import { ArrowLeft, Database, Download, Play, Settings, FileText, Table, Zap } from 'lucide-react';
import '../styles/DataPipelineGenerator.css';

const DataPipelineGenerator = ({ onBack }) => {
  const [activeTab, setActiveTab] = useState('schema');
  const [loading, setLoading] = useState(false);
  const [generatedData, setGeneratedData] = useState(null);
  const [error, setError] = useState('');

  // Schema Definition State
  const [schemaConfig, setSchemaConfig] = useState({
    tableName: 'users',
    recordCount: 1000,
    format: 'csv',
    fields: [
      { name: 'id', type: 'integer', constraints: 'primary_key' },
      { name: 'name', type: 'string', constraints: 'not_null' },
      { name: 'email', type: 'email', constraints: 'unique' },
      { name: 'created_at', type: 'datetime', constraints: '' }
    ]
  });

  // Pipeline Testing State
  const [pipelineConfig, setPipelineConfig] = useState({
    testType: 'etl_validation',
    dataQuality: 'high',
    includeEdgeCases: true,
    includeNulls: false,
    duplicatePercentage: 0
  });

  // Data Types and Constraints
  const dataTypes = [
    'string', 'integer', 'float', 'boolean', 'datetime', 'date', 
    'email', 'phone', 'url', 'uuid', 'json', 'text'
  ];

  const constraints = [
    'not_null', 'unique', 'primary_key', 'foreign_key', 
    'check', 'default', 'auto_increment'
  ];

  const testTypes = [
    { value: 'etl_validation', label: 'ETL Validation Data' },
    { value: 'load_testing', label: 'Load Testing Data' },
    { value: 'data_quality', label: 'Data Quality Testing' },
    { value: 'edge_cases', label: 'Edge Case Testing' },
    { value: 'performance', label: 'Performance Testing' }
  ];

  const formats = [
    { value: 'csv', label: 'CSV' },
    { value: 'json', label: 'JSON' },
    { value: 'parquet', label: 'Parquet' },
    { value: 'sql', label: 'SQL Inserts' }
  ];

  const addField = () => {
    setSchemaConfig(prev => ({
      ...prev,
      fields: [...prev.fields, { name: '', type: 'string', constraints: '' }]
    }));
  };

  const removeField = (index) => {
    setSchemaConfig(prev => ({
      ...prev,
      fields: prev.fields.filter((_, i) => i !== index)
    }));
  };

  const updateField = (index, field, value) => {
    setSchemaConfig(prev => ({
      ...prev,
      fields: prev.fields.map((f, i) => i === index ? { ...f, [field]: value } : f)
    }));
  };

  const generateTestData = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:8000/generate-pipeline-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          schema: schemaConfig,
          pipeline: pipelineConfig
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate test data');
      }

      const result = await response.json();
      setGeneratedData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadData = () => {
    if (!generatedData || !generatedData.data) return;

    const blob = new Blob([generatedData.data], { 
      type: schemaConfig.format === 'json' ? 'application/json' : 'text/csv' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `test_data.${schemaConfig.format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="data-pipeline-container">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="data-pipeline-header">
          <div className="back-navigation">
            <button
              onClick={onBack}
              className="back-navigation"
            >
              <ArrowLeft size={20} />
              Back to Tools
            </button>
          </div>
          
          <div className="header-title-section">
            <div className="header-icon">
              <Database size={24} />
            </div>
            <div className="header-text">
              <h1>Data Pipeline Test Generator</h1>
              <p>Generate realistic test data for data pipeline validation and testing</p>
            </div>
          </div>

          {/* Tabs */}
          <div className="tab-navigation">
            <button
              onClick={() => setActiveTab('schema')}
              className={`tab-button ${activeTab === 'schema' ? 'active' : ''}`}
            >
              <Table size={16} />
              Schema Design
            </button>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`tab-button ${activeTab === 'pipeline' ? 'active' : ''}`}
            >
              <Settings size={16} />
              Pipeline Config
            </button>
            <button
              onClick={() => setActiveTab('generate')}
              className={`tab-button ${activeTab === 'generate' ? 'active' : ''}`}
            >
              <Zap size={16} />
              Generate Data
            </button>
          </div>
        </div>

        {/* Schema Design Tab */}
        {activeTab === 'schema' && (
          <div className="content-card">
            <h2 className="content-title">
              <Table size={20} />
              Define Data Schema
            </h2>
            
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Table Name</label>
                <input
                  type="text"
                  value={schemaConfig.tableName}
                  onChange={(e) => setSchemaConfig(prev => ({ ...prev, tableName: e.target.value }))}
                  className="form-input"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Record Count</label>
                <input
                  type="number"
                  value={schemaConfig.recordCount}
                  onChange={(e) => setSchemaConfig(prev => ({ ...prev, recordCount: parseInt(e.target.value) }))}
                  className="form-input"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Output Format</label>
                <select
                  value={schemaConfig.format}
                  onChange={(e) => setSchemaConfig(prev => ({ ...prev, format: e.target.value }))}
                  className="form-select"
                >
                  {formats.map(format => (
                    <option key={format.value} value={format.value}>{format.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="mb-4">
              <div className="fields-header">
                <h3 className="fields-title">Fields Definition</h3>
                <button
                  onClick={addField}
                  className="add-field-btn"
                >
                  Add Field
                </button>
              </div>
              
              <div className="space-y-3">
                {schemaConfig.fields.map((field, index) => (
                  <div key={index} className="field-row">
                    <input
                      type="text"
                      placeholder="Field name"
                      value={field.name}
                      onChange={(e) => updateField(index, 'name', e.target.value)}
                      className="field-input"
                    />
                    
                    <select
                      value={field.type}
                      onChange={(e) => updateField(index, 'type', e.target.value)}
                      className="field-input"
                    >
                      {dataTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                    
                    <select
                      value={field.constraints}
                      onChange={(e) => updateField(index, 'constraints', e.target.value)}
                      className="field-input"
                    >
                      <option value="">No constraints</option>
                      {constraints.map(constraint => (
                        <option key={constraint} value={constraint}>{constraint}</option>
                      ))}
                    </select>
                    
                    <button
                      onClick={() => removeField(index)}
                      className="remove-field-btn"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Pipeline Config Tab */}
        {activeTab === 'pipeline' && (
          <div className="content-card">
            <h2 className="content-title">
              <Settings size={20} />
              Pipeline Testing Configuration
            </h2>
            
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Test Type</label>
                <select
                  value={pipelineConfig.testType}
                  onChange={(e) => setPipelineConfig(prev => ({ ...prev, testType: e.target.value }))}
                  className="form-select"
                >
                  {testTypes.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>
              
              <div className="form-group">
                <label className="form-label">Data Quality Level</label>
                <select
                  value={pipelineConfig.dataQuality}
                  onChange={(e) => setPipelineConfig(prev => ({ ...prev, dataQuality: e.target.value }))}
                  className="form-select"
                >
                  <option value="high">High Quality</option>
                  <option value="medium">Medium Quality</option>
                  <option value="low">Low Quality (with errors)</option>
                </select>
              </div>
              
              <div className="form-group">
                <label className="form-label">Duplicate Percentage</label>
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={pipelineConfig.duplicatePercentage}
                  onChange={(e) => setPipelineConfig(prev => ({ ...prev, duplicatePercentage: parseInt(e.target.value) }))}
                  className="form-range"
                />
                <span className="text-sm text-gray-600">{pipelineConfig.duplicatePercentage}% duplicates</span>
              </div>
              
              <div className="checkbox-group">
                <label className="checkbox-item">
                  <input
                    type="checkbox"
                    checked={pipelineConfig.includeEdgeCases}
                    onChange={(e) => setPipelineConfig(prev => ({ ...prev, includeEdgeCases: e.target.checked }))}
                    className="form-checkbox"
                  />
                  Include Edge Cases
                </label>
                
                <label className="checkbox-item">
                  <input
                    type="checkbox"
                    checked={pipelineConfig.includeNulls}
                    onChange={(e) => setPipelineConfig(prev => ({ ...prev, includeNulls: e.target.checked }))}
                    className="form-checkbox"
                  />
                  Include NULL Values
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Generate Data Tab */}
        {activeTab === 'generate' && (
          <div className="content-card">
            <h2 className="content-title">
              <Zap size={20} />
              Generate Test Data
            </h2>
            
            <div className="action-section">
              <button
                onClick={generateTestData}
                disabled={loading}
                className="generate-btn"
              >
                <Play size={20} />
                {loading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Generating...
                  </>
                ) : (
                  'Generate Test Data'
                )}
              </button>
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {generatedData && (
              <div className="space-y-4">
                <div className="results-header">
                  <h3 className="results-title">Generated Data Preview</h3>
                  <button
                    onClick={downloadData}
                    className="download-btn"
                  >
                    <Download size={16} />
                    Download
                  </button>
                </div>
                
                <div className="preview-container">
                  <pre className="preview-code">
                    {generatedData.preview}
                  </pre>
                </div>
                
                {generatedData.metadata && (
                  <div className="metadata-grid">
                    <div className="metadata-card blue">
                      <strong>Records Generated:</strong> {generatedData.metadata.recordCount}
                    </div>
                    <div className="metadata-card green">
                      <strong>File Size:</strong> {generatedData.metadata.fileSize}
                    </div>
                    <div className="metadata-card purple">
                      <strong>Generation Time:</strong> {generatedData.metadata.generationTime}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DataPipelineGenerator;
