"""
MCP Client for integrating with the Data Analysis MCP Server
"""
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import io

# Add MCP directory to path
mcp_path = Path(__file__).parent.parent / "MCP"
sys.path.insert(0, str(mcp_path))

# Import the MCP server functions directly
try:
    # Import the functions from the MCP server main.py
    import main as mcp_main
    load_csv_data = mcp_main.load_csv_data
    get_dataset_info = mcp_main.get_dataset_info 
    describe_data = mcp_main.describe_data
    create_visualization = mcp_main.create_visualization
    correlation_analysis = mcp_main.correlation_analysis
    train_ml_model = mcp_main.train_ml_model
    statistical_test = mcp_main.statistical_test
    datasets = mcp_main.datasets
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MCP functions: {e}")
    MCP_AVAILABLE = False
    MCP_AVAILABLE = False


class MCPClient:
    """Client for interacting with the Data Analysis MCP Server"""
    
    def __init__(self):
        self.available = MCP_AVAILABLE
    
    async def load_csv_from_content(self, content: str, filename: str, dataset_name: str = "main") -> Dict[str, Any]:
        """Load CSV data from content string"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Call MCP function directly
            result = load_csv_data(temp_file_path, dataset_name)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return result
        except Exception as e:
            return {"error": f"Failed to load CSV: {str(e)}"}
    
    async def create_visualization(self, 
                                 dataset_name: str = "main",
                                 plot_type: str = "scatter",
                                 x_column: str = "",
                                 y_column: Optional[str] = None,
                                 color_column: Optional[str] = None,
                                 title: Optional[str] = None) -> Any:
        """Create visualization using MCP server"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            # Call MCP function directly
            result = create_visualization(
                dataset_name=dataset_name,
                plot_type=plot_type,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                title=title
            )
            return result
        except Exception as e:
            return {"error": f"Failed to create visualization: {str(e)}"}
    
    async def get_dataset_info(self, dataset_name: str = "main") -> Dict[str, Any]:
        """Get dataset information"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = get_dataset_info(dataset_name)
            return result
        except Exception as e:
            return {"error": f"Failed to get dataset info: {str(e)}"}
    
    async def describe_data(self, dataset_name: str = "main", include_all: bool = False) -> Dict[str, Any]:
        """Get descriptive statistics"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = describe_data(dataset_name, include_all)
            return result
        except Exception as e:
            return {"error": f"Failed to describe data: {str(e)}"}
    
    async def correlation_analysis(self, dataset_name: str = "main", method: str = "pearson") -> Dict[str, Any]:
        """Perform correlation analysis"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = correlation_analysis(dataset_name, method)
            return result
        except Exception as e:
            return {"error": f"Failed to perform correlation analysis: {str(e)}"}
    
    async def train_ml_model(self,
                           dataset_name: str = "main",
                           target_column: str = "",
                           model_type: str = "linear_regression",
                           test_size: float = 0.2,
                           features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train machine learning model"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = train_ml_model(
                dataset_name=dataset_name,
                target_column=target_column,
                model_type=model_type,
                test_size=test_size,
                features=features
            )
            return result
        except Exception as e:
            return {"error": f"Failed to train ML model: {str(e)}"}
    
    async def statistical_test(self,
                             dataset_name: str = "main",
                             test_type: str = "ttest",
                             column1: str = "",
                             column2: Optional[str] = None,
                             groupby: Optional[str] = None) -> Dict[str, Any]:
        """Perform statistical tests"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = statistical_test(
                dataset_name=dataset_name,
                test_type=test_type,
                column1=column1,
                column2=column2,
                groupby=groupby
            )
            return result
        except Exception as e:
            return {"error": f"Failed to perform statistical test: {str(e)}"}


# Global MCP client instance
mcp_client = MCPClient()
