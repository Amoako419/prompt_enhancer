"""
Data Analysis MCP Server

A comprehensive Model Context Protocol server that provides powerful data analysis capabilities
including data processing, statistical analysis, visualization, and machine learning tools.

This server enables AI applications to:
- Load and process various data formats (CSV, Excel, JSON)
- Perform statistical analysis and hypothesis testing
- Generate visualizations and plots
- Apply machine learning algorithms
- Export results and reports

Key Features:
- FastMCP framework for easy MCP server development
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning
- Matplotlib, Seaborn, and Plotly for visualization
- Statistical analysis with SciPy and Statsmodels
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP and FastAPI imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.utilities.types import Image

# Data analysis imports
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
mcp = FastMCP(
    name="DataAnalysisMCP",
    instructions="""
    A comprehensive data analysis server that provides tools for:
    
    1. Data Loading & Processing:
       - Load CSV, Excel, JSON files
       - Clean and preprocess data
       - Handle missing values
       
    2. Statistical Analysis:
       - Descriptive statistics
       - Hypothesis testing
       - Correlation analysis
       
    3. Visualization:
       - Generate various plot types
       - Create interactive charts
       - Export visualizations
       
    4. Machine Learning:
       - Regression and classification
       - Clustering analysis
       - Model evaluation
       
    Use these tools to perform comprehensive data analysis workflows.
    """,
    dependencies=[
        "pandas>=2.1.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "openpyxl>=3.1.0"
    ]
)

# Global storage for datasets
datasets: Dict[str, pd.DataFrame] = {}

@mcp.tool()
def load_csv_data(file_path: str, dataset_name: str = "main", **kwargs) -> Dict[str, Any]:
    """
    Load CSV data into memory for analysis.
    
    Args:
        file_path: Path to the CSV file
        dataset_name: Name to store the dataset under (default: "main")
        **kwargs: Additional pandas read_csv parameters
        
    Returns:
        Dictionary with dataset info and preview
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, **kwargs)
        datasets[dataset_name] = df
        
        return {
            "message": f"Successfully loaded CSV data as '{dataset_name}'",
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "preview": df.head().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        return {"error": f"Failed to load CSV: {str(e)}"}

@mcp.tool()
def load_csv_from_content(content: str, filename: str, dataset_name: str = "main", **kwargs) -> Dict[str, Any]:
    """
    Load CSV data from string content into memory for analysis.
    
    Args:
        content: CSV content as string
        filename: Original filename for reference
        dataset_name: Name to store the dataset under (default: "main")
        **kwargs: Additional pandas read_csv parameters
        
    Returns:
        Dictionary with dataset info and preview
    """
    try:
        # Create a StringIO object from the content
        from io import StringIO
        csv_io = StringIO(content)
        
        # Try to detect if the first row contains headers
        # Read first few lines to analyze
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return {"error": "CSV file must contain at least 2 rows"}
        
        # Improved header detection
        first_row = lines[0].split(',')
        second_row = lines[1].split(',') if len(lines) > 1 else []
        
        # Multiple criteria for header detection
        has_headers = False
        
        # Criterion 1: Check if first row contains non-numeric strings
        non_numeric_in_first = any(
            not cell.strip().replace('.', '').replace('-', '').replace('+', '').isdigit() 
            and cell.strip() != '' 
            for cell in first_row
        )
        
        # Criterion 2: Check if first row contains common header-like words
        header_words = ['id', 'name', 'date', 'time', 'sales', 'quantity', 'price', 'amount', 'total', 'count']
        contains_header_words = any(
            any(word in cell.strip().lower() for word in header_words)
            for cell in first_row
        )
        
        # Criterion 3: If second row is more numeric than first row
        if len(second_row) == len(first_row):
            first_row_numeric_count = sum(
                1 for cell in first_row 
                if cell.strip().replace('.', '').replace('-', '').isdigit()
            )
            second_row_numeric_count = sum(
                1 for cell in second_row 
                if cell.strip().replace('.', '').replace('-', '').isdigit()
            )
            second_more_numeric = second_row_numeric_count > first_row_numeric_count
        else:
            second_more_numeric = False
        
        # Final decision: headers detected if any criterion is met
        has_headers = non_numeric_in_first or contains_header_words or second_more_numeric
        
        # Reset StringIO and load the CSV file with appropriate header setting
        csv_io = StringIO(content)
        if has_headers:
            df = pd.read_csv(csv_io, **kwargs)
        else:
            df = pd.read_csv(csv_io, header=None, **kwargs)
        
        # Store the dataset
        datasets[dataset_name] = df
        
        # Get comprehensive dataset info
        dataset_info = get_dataset_info(dataset_name)
        
        return {
            "message": f"Successfully loaded CSV data from '{filename}' as '{dataset_name}'",
            "filename": filename,
            "has_headers": has_headers,
            "shape": list(df.shape),
            "columns": [str(col) for col in df.columns],
            "data": dataset_info.get("data", {}),
            "columns_info": dataset_info.get("columns", {}),
            "preview": dataset_info.get("preview", []),
            "memory_usage": dataset_info.get("memory_usage", 0)
        }
    except Exception as e:
        return {"error": f"Failed to load CSV from content: {str(e)}"}

@mcp.tool()
def load_excel_data(file_path: str, sheet_name: Optional[str] = None, dataset_name: str = "main", **kwargs) -> Dict[str, Any]:
    """
    Load Excel data into memory for analysis.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to load (default: first sheet)
        dataset_name: Name to store the dataset under
        **kwargs: Additional pandas read_excel parameters
        
    Returns:
        Dictionary with dataset info and preview
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        datasets[dataset_name] = df
        
        return {
            "message": f"Successfully loaded Excel data as '{dataset_name}'",
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "preview": df.head().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        return {"error": f"Failed to load Excel: {str(e)}"}

@mcp.tool()
def get_dataset_info(dataset_name: str = "main") -> Dict[str, Any]:
    """
    Get comprehensive information about a loaded dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
        
    Returns:
        Dictionary with detailed dataset information
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found. Available datasets: {list(datasets.keys())}"}
    
    df = datasets[dataset_name]
    
    try:
        # Build comprehensive column information
        columns_info = {}
        for col in df.columns:
            col_data = df[col]
            
            # Get basic type information
            dtype_str = str(col_data.dtype)
            
            # Get non-null count
            non_null_count = col_data.notna().sum()
            
            # Get sample values (first few non-null unique values)
            sample_values = col_data.dropna().unique()[:5].tolist()
            
            # Convert numpy types to native Python types for JSON serialization
            sample_values = [
                int(x) if isinstance(x, (np.integer, np.int64)) else
                float(x) if isinstance(x, (np.floating, np.float64)) else
                str(x) for x in sample_values
            ]
            
            columns_info[str(col)] = {
                "dtype": dtype_str,
                "non_null_count": int(non_null_count),
                "sample_values": sample_values,
                "total_count": len(col_data),
                "null_count": int(col_data.isna().sum()),
                "unique_count": int(col_data.nunique())
            }
        
        # Get preview data with proper column names
        preview_data = []
        for _, row in df.head().iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                # Convert numpy types to native Python types
                if pd.isna(value):
                    row_dict[str(col)] = None
                elif isinstance(value, (np.integer, np.int64)):
                    row_dict[str(col)] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    row_dict[str(col)] = float(value)
                else:
                    row_dict[str(col)] = str(value)
            preview_data.append(row_dict)
        
        info = {
            "name": dataset_name,
            "shape": list(df.shape),  # Convert to list for JSON serialization
            "columns": columns_info,
            "preview": preview_data,
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "missing_values": {str(col): int(count) for col, count in df.isnull().sum().items() if count > 0},
            "numeric_columns": [str(col) for col in df.select_dtypes(include=[np.number]).columns],
            "categorical_columns": [str(col) for col in df.select_dtypes(include=['object', 'category']).columns],
            "datetime_columns": [str(col) for col in df.select_dtypes(include=['datetime']).columns]
        }
            
        return info
    except Exception as e:
        return {"error": f"Failed to get dataset info: {str(e)}"}

@mcp.tool()
def describe_data(dataset_name: str = "main", include_all: bool = False) -> Dict[str, Any]:
    """
    Generate descriptive statistics for the dataset.
    
    Args:
        dataset_name: Name of the dataset to describe
        include_all: Include statistics for all columns, not just numeric
        
    Returns:
        Dictionary with descriptive statistics
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        if include_all:
            description = df.describe(include='all').fillna('').to_dict()
        else:
            description = df.describe().to_dict()
            
        return {
            "dataset": dataset_name,
            "statistics": description,
            "shape": df.shape
        }
    except Exception as e:
        return {"error": f"Failed to describe data: {str(e)}"}

@mcp.tool()
def clean_data(dataset_name: str = "main", 
               drop_duplicates: bool = False,
               fill_missing: Optional[str] = None,
               drop_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Clean the dataset by handling missing values, duplicates, etc.
    
    Args:
        dataset_name: Name of the dataset to clean
        drop_duplicates: Whether to drop duplicate rows
        fill_missing: Strategy for missing values ('mean', 'median', 'mode', 'forward', 'backward')
        drop_columns: List of columns to drop
        
    Returns:
        Dictionary with cleaning results
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name].copy()
    changes = []
    
    try:
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
            changes.append(f"Dropped columns: {drop_columns}")
        
        # Handle duplicates
        if drop_duplicates:
            before_count = len(df)
            df = df.drop_duplicates()
            after_count = len(df)
            changes.append(f"Removed {before_count - after_count} duplicate rows")
        
        # Handle missing values
        if fill_missing:
            missing_before = df.isnull().sum().sum()
            
            if fill_missing == 'mean':
                df = df.fillna(df.select_dtypes(include=[np.number]).mean())
            elif fill_missing == 'median':
                df = df.fillna(df.select_dtypes(include=[np.number]).median())
            elif fill_missing == 'mode':
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')
            elif fill_missing == 'forward':
                df = df.fillna(method='ffill')
            elif fill_missing == 'backward':
                df = df.fillna(method='bfill')
            
            missing_after = df.isnull().sum().sum()
            changes.append(f"Filled {missing_before - missing_after} missing values using {fill_missing}")
        
        # Update the stored dataset
        datasets[dataset_name] = df
        
        return {
            "message": f"Successfully cleaned dataset '{dataset_name}'",
            "changes": changes,
            "new_shape": df.shape,
            "remaining_nulls": df.isnull().sum().sum()
        }
    except Exception as e:
        return {"error": f"Failed to clean data: {str(e)}"}

@mcp.tool()
def correlation_analysis(dataset_name: str = "main", method: str = "pearson") -> Dict[str, Any]:
    """
    Perform correlation analysis on numeric columns.
    
    Args:
        dataset_name: Name of the dataset to analyze
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Dictionary with correlation matrix and insights
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}
    
    try:
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        "variables": [corr_matrix.columns[i], corr_matrix.columns[j]],
                        "correlation": corr_val
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_corr,
            "method": method,
            "numeric_columns": numeric_cols.tolist()
        }
    except Exception as e:
        return {"error": f"Failed to perform correlation analysis: {str(e)}"}

@mcp.tool()
def statistical_test(dataset_name: str = "main", 
                     test_type: str = "ttest",
                     column1: str = "",
                     column2: Optional[str] = None,
                     groupby: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform statistical hypothesis tests.
    
    Args:
        dataset_name: Name of the dataset
        test_type: Type of test ('ttest', 'chi2', 'anova', 'normality')
        column1: First column for analysis
        column2: Second column (if needed)
        groupby: Column to group by (for grouped tests)
        
    Returns:
        Dictionary with test results
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        result = {"test_type": test_type}
        
        if test_type == "ttest":
            if column2:
                # Two-sample t-test
                stat, pvalue = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
                result.update({
                    "statistic": stat,
                    "p_value": pvalue,
                    "interpretation": "Significant difference" if pvalue < 0.05 else "No significant difference"
                })
            elif groupby:
                # Grouped t-test
                groups = df.groupby(groupby)[column1].apply(list)
                if len(groups) == 2:
                    stat, pvalue = stats.ttest_ind(groups.iloc[0], groups.iloc[1])
                    result.update({
                        "statistic": stat,
                        "p_value": pvalue,
                        "groups": groups.index.tolist(),
                        "interpretation": "Significant difference" if pvalue < 0.05 else "No significant difference"
                    })
        
        elif test_type == "normality":
            # Shapiro-Wilk normality test
            stat, pvalue = stats.shapiro(df[column1].dropna())
            result.update({
                "statistic": stat,
                "p_value": pvalue,
                "interpretation": "Not normally distributed" if pvalue < 0.05 else "Normally distributed"
            })
        
        elif test_type == "chi2" and column2:
            # Chi-square test of independence
            contingency = pd.crosstab(df[column1], df[column2])
            stat, pvalue, dof, expected = stats.chi2_contingency(contingency)
            result.update({
                "statistic": stat,
                "p_value": pvalue,
                "degrees_of_freedom": dof,
                "contingency_table": contingency.to_dict(),
                "interpretation": "Variables are dependent" if pvalue < 0.05 else "Variables are independent"
            })
        
        return result
    except Exception as e:
        return {"error": f"Failed to perform statistical test: {str(e)}"}

@mcp.tool()
def create_visualization(dataset_name: str = "main",
                        plot_type: str = "scatter",
                        x_column: str = "",
                        y_column: Optional[str] = None,
                        color_column: Optional[str] = None,
                        title: Optional[str] = None) -> Image:
    """
    Create various types of data visualizations.
    
    Args:
        dataset_name: Name of the dataset
        plot_type: Type of plot ('scatter', 'line', 'bar', 'histogram', 'box', 'heatmap')
        x_column: Column for x-axis
        y_column: Column for y-axis (if needed)
        color_column: Column for color coding
        title: Plot title
        
    Returns:
        Image object containing the plot
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        plt.figure(figsize=(10, 6))
        
        if plot_type == "scatter" and y_column:
            if color_column:
                scatter = plt.scatter(df[x_column], df[y_column], c=df[color_column], alpha=0.6)
                plt.colorbar(scatter)
            else:
                plt.scatter(df[x_column], df[y_column], alpha=0.6)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif plot_type == "line" and y_column:
            plt.plot(df[x_column], df[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif plot_type == "bar":
            if y_column:
                plt.bar(df[x_column], df[y_column])
                plt.ylabel(y_column)
            else:
                df[x_column].value_counts().plot(kind='bar')
            plt.xlabel(x_column)
            
        elif plot_type == "histogram":
            plt.hist(df[x_column], bins=30, alpha=0.7)
            plt.xlabel(x_column)
            plt.ylabel("Frequency")
            
        elif plot_type == "box":
            if y_column:
                df.boxplot(column=y_column, by=x_column)
            else:
                plt.boxplot(df[x_column].dropna())
                plt.ylabel(x_column)
                
        elif plot_type == "heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{plot_type.title()} Plot")
        
        plt.tight_layout()
        
        # Save plot to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        # Create Image object
        image_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()  # Close the figure to free memory
        
        return Image(data=image_data, media_type="image/png")
        
    except Exception as e:
        plt.close()  # Ensure we close the figure even on error
        return {"error": f"Failed to create visualization: {str(e)}"}

@mcp.tool() 
def train_ml_model(dataset_name: str = "main",
                   target_column: str = "",
                   model_type: str = "linear_regression",
                   test_size: float = 0.2,
                   features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Train a machine learning model on the dataset.
    
    Args:
        dataset_name: Name of the dataset
        target_column: Column to predict
        model_type: Type of model ('linear_regression', 'logistic_regression', 'random_forest', 'kmeans')
        test_size: Fraction of data to use for testing
        features: List of feature columns (if None, uses all numeric columns except target)
        
    Returns:
        Dictionary with model results and evaluation metrics
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        # Handle feature selection
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            features = numeric_cols
        
        if not features:
            return {"error": "No valid features found"}
        
        # Prepare data
        X = df[features].dropna()
        
        if model_type == "kmeans":
            # Clustering doesn't need target
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = KMeans(n_clusters=3, random_state=42)
            clusters = model.fit_predict(X_scaled)
            
            return {
                "model_type": model_type,
                "n_clusters": 3,
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia": model.inertia_,
                "cluster_assignments": clusters.tolist(),
                "features_used": features
            }
        
        # For supervised learning
        y = df[target_column].loc[X.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if model_type == "linear_regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2_score = model.score(X_test, y_test)
            
            return {
                "model_type": model_type,
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2_score": r2_score,
                "coefficients": dict(zip(features, model.coef_)),
                "intercept": model.intercept_,
                "features_used": features
            }
            
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                "model_type": model_type,
                "accuracy": accuracy,
                "coefficients": dict(zip(features, model.coef_[0])),
                "intercept": model.intercept_[0],
                "features_used": features,
                "unique_classes": model.classes_.tolist()
            }
            
        elif model_type == "random_forest":
            # Determine if regression or classification based on target
            if df[target_column].dtype in ['object', 'category'] or len(df[target_column].unique()) < 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                return {
                    "model_type": f"{model_type}_classifier",
                    "accuracy": accuracy,
                    "feature_importance": dict(zip(features, model.feature_importances_)),
                    "features_used": features,
                    "n_estimators": 100
                }
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2_score = model.score(X_test, y_test)
                
                return {
                    "model_type": f"{model_type}_regressor",
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "r2_score": r2_score,
                    "feature_importance": dict(zip(features, model.feature_importances_)),
                    "features_used": features,
                    "n_estimators": 100
                }
        
    except Exception as e:
        return {"error": f"Failed to train model: {str(e)}"}

@mcp.tool()
def export_data(dataset_name: str = "main", 
               file_path: str = "",
               file_format: str = "csv") -> Dict[str, Any]:
    """
    Export dataset to various file formats.
    
    Args:
        dataset_name: Name of the dataset to export
        file_path: Path where to save the file
        file_format: Format to export ('csv', 'excel', 'json')
        
    Returns:
        Dictionary with export results
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        if file_format == "csv":
            df.to_csv(file_path, index=False)
        elif file_format == "excel":
            df.to_excel(file_path, index=False)
        elif file_format == "json":
            df.to_json(file_path, orient='records', indent=2)
        else:
            return {"error": f"Unsupported format: {file_format}"}
        
        return {
            "message": f"Successfully exported dataset '{dataset_name}' to {file_path}",
            "format": file_format,
            "shape": df.shape,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
    except Exception as e:
        return {"error": f"Failed to export data: {str(e)}"}

@mcp.tool()
def list_datasets() -> Dict[str, Any]:
    """
    List all currently loaded datasets.
    
    Returns:
        Dictionary with information about loaded datasets
    """
    dataset_info = {}
    for name, df in datasets.items():
        dataset_info[name] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_count": df.isnull().sum().sum()
        }
    
    return {
        "datasets": dataset_info,
        "total_datasets": len(datasets)
    }

@mcp.tool()
def query_data(dataset_name: str = "main", 
               query: str = "") -> Dict[str, Any]:
    """
    Query dataset using pandas query syntax.
    
    Args:
        dataset_name: Name of the dataset to query
        query: Pandas query string (e.g., "age > 25 and income < 50000")
        
    Returns:
        Dictionary with query results
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        result_df = df.query(query)
        
        return {
            "query": query,
            "original_shape": df.shape,
            "result_shape": result_df.shape,
            "result_preview": result_df.head(10).to_dict(),
            "matched_rows": len(result_df)
        }
    except Exception as e:
        return {"error": f"Failed to execute query: {str(e)}"}

def suggest_chart_types(dataset_name: str = "main") -> Dict[str, Any]:
    """
    Analyze dataset characteristics and suggest appropriate chart types.
    
    Args:
        dataset_name: Name of the dataset to analyze
        
    Returns:
        Dictionary with chart recommendations and explanations
    """
    if dataset_name not in datasets:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    df = datasets[dataset_name]
    
    try:
        # Analyze data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        num_rows, num_cols = df.shape
        num_numeric = len(numeric_cols)
        num_categorical = len(categorical_cols)
        num_datetime = len(datetime_cols)
        
        recommendations = []
        
        # Basic data insights
        insights = {
            "total_columns": num_cols,
            "total_rows": num_rows,
            "numeric_columns": num_numeric,
            "categorical_columns": num_categorical,
            "datetime_columns": num_datetime,
            "numeric_column_names": numeric_cols,
            "categorical_column_names": categorical_cols,
            "datetime_column_names": datetime_cols
        }
        
        # Chart recommendations based on data characteristics
        
        # 1. Histograms for numeric distributions
        if num_numeric > 0:
            recommendations.append({
                "chart_type": "histogram",
                "priority": "high",
                "reason": f"Great for exploring the distribution of numeric variables ({num_numeric} numeric columns available)",
                "best_for": "Understanding data distribution, identifying outliers, checking normality",
                "suitable_columns": numeric_cols[:3]  # Limit to first 3 for display
            })
        
        # 2. Correlation heatmap for multiple numeric variables
        if num_numeric >= 2:
            recommendations.append({
                "chart_type": "correlation_heatmap",
                "priority": "high",
                "reason": f"Excellent for identifying relationships between {num_numeric} numeric variables",
                "best_for": "Finding correlations, feature selection, understanding variable relationships",
                "suitable_columns": numeric_cols
            })
        
        # 3. Bar charts for categorical data
        if num_categorical > 0:
            recommendations.append({
                "chart_type": "bar",
                "priority": "high",
                "reason": f"Perfect for comparing categories in {num_categorical} categorical column(s)",
                "best_for": "Comparing category frequencies, identifying dominant categories",
                "suitable_columns": categorical_cols[:2]  # Limit for display
            })
        
        # 4. Scatter plots for numeric relationships
        if num_numeric >= 2:
            recommendations.append({
                "chart_type": "scatter",
                "priority": "medium",
                "reason": f"Useful for exploring relationships between pairs of numeric variables",
                "best_for": "Identifying patterns, trends, and outliers in relationships",
                "suitable_columns": numeric_cols[:2]
            })
        
        # 5. Box plots for distribution and outliers
        if num_numeric > 0:
            recommendations.append({
                "chart_type": "box",
                "priority": "medium",
                "reason": f"Excellent for identifying outliers and distribution characteristics",
                "best_for": "Outlier detection, comparing distributions across groups",
                "suitable_columns": numeric_cols[:3]
            })
        
        # 6. Time series plots for datetime data
        if num_datetime > 0 and num_numeric > 0:
            recommendations.append({
                "chart_type": "line",
                "priority": "high",
                "reason": f"Essential for time-based data analysis with {num_datetime} datetime column(s)",
                "best_for": "Trend analysis, seasonal patterns, time-based insights",
                "suitable_columns": datetime_cols + numeric_cols[:2]
            })
        
        # 7. Pie charts (only recommend for small categorical sets)
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 6:  # Only recommend for small category sets
                recommendations.append({
                    "chart_type": "pie",
                    "priority": "low",
                    "reason": f"Suitable for '{col}' with {unique_count} categories (small categorical set)",
                    "best_for": "Showing proportions of a whole, part-to-whole relationships",
                    "suitable_columns": [col]
                })
        
        # Sort recommendations by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        # Limit to top 6 recommendations to avoid overwhelming the user
        recommendations = recommendations[:6]
        
        return {
            "dataset_info": insights,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "analysis_summary": f"Based on your dataset with {num_rows} rows and {num_cols} columns ({num_numeric} numeric, {num_categorical} categorical, {num_datetime} datetime), we recommend starting with the high-priority charts."
        }
        
    except Exception as e:
        return {"error": f"Failed to analyze dataset for chart recommendations: {str(e)}"}

@mcp.tool()
async def get_chart_recommendations(dataset_name: str = "main") -> Dict[str, Any]:
    """
    Get intelligent chart recommendations based on dataset characteristics.
    
    This tool analyzes your dataset and suggests the most appropriate chart types
    based on the data types, number of variables, and data characteristics.
    
    Args:
        dataset_name: Name of the dataset to analyze (default: "main")
        
    Returns:
        Dictionary containing:
        - dataset_info: Basic information about the dataset
        - recommendations: List of recommended charts with explanations
        - analysis_summary: Overall recommendation summary
    """
    return suggest_chart_types(dataset_name)

# Run the server
if __name__ == "__main__":
    mcp.run()
