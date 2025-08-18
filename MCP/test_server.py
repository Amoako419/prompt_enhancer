"""
Test script for the Data Analysis MCP Server.

This script demonstrates how to use the various tools provided by the server.
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import the server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import mcp
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample datasets for testing."""
    # Sample sales data
    np.random.seed(42)
    n_samples = 1000
    
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Sports'], n_samples),
        'advertising_spend': np.random.uniform(1000, 10000, n_samples),
        'revenue': np.random.uniform(5000, 50000, n_samples),
        'units_sold': np.random.randint(10, 500, n_samples),
        'customer_age': np.random.randint(18, 80, n_samples),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples)
    })
    
    # Add some correlation between advertising and revenue
    sales_data['revenue'] = sales_data['advertising_spend'] * 3.5 + np.random.normal(0, 5000, n_samples)
    sales_data['revenue'] = np.clip(sales_data['revenue'], 5000, 50000)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    sales_data.to_csv('data/sample_sales.csv', index=False)
    
    # Sample customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 501),
        'age': np.random.randint(18, 80, 500),
        'income': np.random.uniform(30000, 120000, 500),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
        'purchase_amount': np.random.uniform(100, 5000, 500),
        'loyalty_score': np.random.uniform(0, 10, 500),
        'churn': np.random.choice([0, 1], 500, p=[0.7, 0.3])
    })
    
    customer_data.to_csv('data/sample_customers.csv', index=False)
    
    print("Sample data created successfully!")
    print("Files created:")
    print("- data/sample_sales.csv")
    print("- data/sample_customers.csv")

async def test_data_analysis_tools():
    """Test the data analysis tools."""
    print("\n" + "="*50)
    print("TESTING DATA ANALYSIS MCP SERVER")
    print("="*50)
    
    # Create sample data first
    create_sample_data()
    
    # Test data loading
    print("\n1. Testing data loading...")
    
    # Load CSV data
    result = await mcp.call_tool("load_csv_data", {
        "file_path": "data/sample_sales.csv",
        "dataset_name": "sales"
    })
    print("Load CSV result:", result)
    
    # Load customer data
    result = await mcp.call_tool("load_csv_data", {
        "file_path": "data/sample_customers.csv", 
        "dataset_name": "customers"
    })
    print("Load customers result:", result)
    
    # Test dataset info
    print("\n2. Testing dataset information...")
    result = await mcp.call_tool("get_dataset_info", {"dataset_name": "sales"})
    print("Dataset info:", result)
    
    # Test descriptive statistics
    print("\n3. Testing descriptive statistics...")
    result = await mcp.call_tool("describe_data", {"dataset_name": "sales"})
    print("Descriptive stats:", result)
    
    # Test correlation analysis
    print("\n4. Testing correlation analysis...")
    result = await mcp.call_tool("correlation_analysis", {
        "dataset_name": "sales",
        "method": "pearson"
    })
    print("Correlation analysis:", result)
    
    # Test data cleaning
    print("\n5. Testing data cleaning...")
    result = await mcp.call_tool("clean_data", {
        "dataset_name": "sales",
        "drop_duplicates": True,
        "fill_missing": "mean"
    })
    print("Data cleaning result:", result)
    
    # Test statistical test
    print("\n6. Testing statistical test...")
    result = await mcp.call_tool("statistical_test", {
        "dataset_name": "sales",
        "test_type": "ttest",
        "column1": "revenue",
        "groupby": "region"
    })
    print("Statistical test result:", result)
    
    # Test machine learning
    print("\n7. Testing machine learning...")
    result = await mcp.call_tool("train_ml_model", {
        "dataset_name": "sales",
        "target_column": "revenue",
        "model_type": "linear_regression",
        "features": ["advertising_spend", "units_sold", "customer_age"]
    })
    print("ML model result:", result)
    
    # Test clustering
    print("\n8. Testing clustering...")
    result = await mcp.call_tool("train_ml_model", {
        "dataset_name": "customers",
        "model_type": "kmeans"
    })
    print("Clustering result:", result)
    
    # Test query data
    print("\n9. Testing data querying...")
    result = await mcp.call_tool("query_data", {
        "dataset_name": "sales",
        "query": "revenue > 30000 and region == 'North'"
    })
    print("Query result:", result)
    
    # Test list datasets
    print("\n10. Testing list datasets...")
    result = await mcp.call_tool("list_datasets", {})
    print("List datasets result:", result)
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(test_data_analysis_tools())
