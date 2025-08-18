# Data Analysis MCP Server

A comprehensive Model Context Protocol (MCP) server that provides powerful data analysis capabilities using FastMCP and various Python data science libraries.

## Features

### 🔧 Data Loading & Processing
- Load CSV, Excel, and JSON files
- Clean and preprocess data
- Handle missing values and duplicates
- Query data using pandas syntax

### 📊 Statistical Analysis  
- Descriptive statistics
- Correlation analysis
- Hypothesis testing (t-tests, chi-square, ANOVA)
- Normality testing

### 📈 Data Visualization
- Multiple plot types (scatter, line, bar, histogram, box plots, heatmaps)
- Color coding and customization
- High-quality image outputs

### 🤖 Machine Learning
- Linear and logistic regression
- Random Forest (classification/regression)
- K-means clustering
- Model evaluation metrics
- Feature importance analysis

### 💾 Data Export
- Export processed data to CSV, Excel, or JSON
- Save visualizations as images

## Installation

1. **Clone/Download the project**
```bash
cd MCP
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

#### Option 1: Direct Python execution
```bash
python main.py
```

#### Option 2: With VS Code MCP extension
The server can be configured in VS Code using the `.vscode/mcp.json` configuration file.

### Available Tools

#### Data Loading
- `load_csv_data(file_path, dataset_name, **kwargs)` - Load CSV files
- `load_excel_data(file_path, sheet_name, dataset_name, **kwargs)` - Load Excel files

#### Data Exploration
- `get_dataset_info(dataset_name)` - Get comprehensive dataset information
- `describe_data(dataset_name, include_all)` - Generate descriptive statistics
- `list_datasets()` - List all loaded datasets
- `query_data(dataset_name, query)` - Query data using pandas syntax

#### Data Cleaning
- `clean_data(dataset_name, drop_duplicates, fill_missing, drop_columns)` - Clean and preprocess data

#### Statistical Analysis
- `correlation_analysis(dataset_name, method)` - Correlation analysis
- `statistical_test(dataset_name, test_type, column1, column2, groupby)` - Statistical hypothesis tests

#### Visualization
- `create_visualization(dataset_name, plot_type, x_column, y_column, color_column, title)` - Create various plots

#### Machine Learning
- `train_ml_model(dataset_name, target_column, model_type, test_size, features)` - Train ML models

#### Data Export
- `export_data(dataset_name, file_path, file_format)` - Export data to files

## Example Workflow

1. **Load Data**
```python
# Load a CSV file
load_csv_data("data/sales.csv", "sales_data")
```

2. **Explore Data**
```python
# Get dataset information
get_dataset_info("sales_data")

# Generate descriptive statistics
describe_data("sales_data")
```

3. **Clean Data**
```python
# Clean the data
clean_data("sales_data", drop_duplicates=True, fill_missing="mean")
```

4. **Analyze**
```python
# Correlation analysis
correlation_analysis("sales_data", method="pearson")

# Statistical test
statistical_test("sales_data", test_type="ttest", column1="revenue", groupby="region")
```

5. **Visualize**
```python
# Create scatter plot
create_visualization("sales_data", "scatter", "advertising_spend", "revenue", "region")
```

6. **Machine Learning**
```python
# Train a model
train_ml_model("sales_data", target_column="revenue", model_type="random_forest")
```

## Configuration

### MCP Configuration (.vscode/mcp.json)
```json
{
  "servers": {
    "data-analysis": {
      "command": "python",
      "args": ["main.py"],
      "env": {}
    }
  }
}
```

### Environment Variables
You can set environment variables for configuration:
- `DATA_ANALYSIS_LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Dependencies

### Core MCP
- `mcp[cli]>=1.2.0` - Model Context Protocol framework

### Data Analysis
- `pandas>=2.1.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.11.0` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning
- `statsmodels>=0.14.0` - Statistical analysis

### Visualization
- `matplotlib>=3.7.0` - Plotting library
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.17.0` - Interactive plots

### File Support
- `openpyxl>=3.1.0` - Excel file support
- `xlrd>=2.0.1` - Excel reading

## Supported File Formats

### Input
- **CSV** - Comma-separated values
- **Excel** - .xlsx, .xls files
- **JSON** - JavaScript Object Notation

### Output
- **CSV** - For processed data
- **Excel** - For reports and processed data
- **JSON** - For structured data export
- **PNG** - For visualizations

## Error Handling

The server includes comprehensive error handling:
- Invalid file paths or formats
- Missing datasets
- Invalid column names
- Insufficient data for analysis
- Model training errors

All functions return structured responses with either results or error messages.

## Performance Considerations

- Large datasets are handled efficiently using pandas
- Memory usage information is provided for loaded datasets
- Visualizations are optimized for quality and performance
- Model training includes progress indicators

## Logging

The server uses Python's logging module for debugging and monitoring:
- Configure log level via environment variables
- Detailed error messages for troubleshooting
- Performance metrics for operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the error messages in the function responses
2. Verify file paths and dataset names
3. Ensure data types are compatible with requested operations
4. Review the comprehensive tool documentation
