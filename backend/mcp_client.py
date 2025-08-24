"""MCP Client for bridging FastAPI with the real MCP server functions."""

import sys
import os
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class MCPClient:
    """Client for interacting with the Data Analysis MCP Server"""
    
    def __init__(self):
        self.mcp_path = Path(__file__).parent.parent / "MCP"
        self.available = self.mcp_path.exists()
    
    def _call_mcp_function(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """Call an MCP function using subprocess to avoid circular imports."""
        if not self.available:
            return {"error": "MCP server not available"}
        
        # Prepare arguments for JSON serialization
        json_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                json_kwargs[key] = value
            else:
                json_kwargs[key] = str(value)
        
        # Create a script to call the MCP function
        script_content = f'''
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add MCP path
mcp_path = Path(r"{self.mcp_path}")
sys.path.insert(0, str(mcp_path))

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # Handle numpy dtypes and other numpy objects
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)

try:
    import main as mcp_main
    
    # Parse arguments
    kwargs = {json.dumps(json_kwargs)}
    
    # Call the function
    result = getattr(mcp_main, "{function_name}")(**kwargs)
    
    # Return success response using custom encoder
    response = {{"success": True, "result": result}}
    print(json.dumps(response, cls=NumpyEncoder))
    
except Exception as e:
    response = {{"success": False, "error": str(e)}}
    print(json.dumps(response, cls=NumpyEncoder))
'''
        
        # Write and execute the script
        temp_script = self.mcp_path / "temp_mcp_call.py"
        try:
            with open(temp_script, 'w') as f:
                f.write(script_content)
            
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True,
                text=True,
                cwd=str(self.mcp_path)
            )
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout.strip())
                    if response.get("success"):
                        return response["result"]
                    else:
                        return {"error": response.get("error", "Unknown error")}
                except json.JSONDecodeError:
                    return {"error": f"Invalid JSON response: {result.stdout}"}
            else:
                return {"error": f"MCP call failed: {result.stderr}"}
        
        finally:
            # Clean up temp file
            if temp_script.exists():
                temp_script.unlink()
    
    async def load_csv_from_content(self, content: str, filename: str, dataset_name: str = "main") -> Dict[str, Any]:
        """Load CSV data from content string"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Call MCP function
            result = self._call_mcp_function(
                "load_csv_data",
                file_path=temp_file_path,
                dataset_name=dataset_name
            )
            
            # Clean up
            os.unlink(temp_file_path)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to load CSV: {str(e)}"}
    
    async def create_visualization(self, dataset_name: str = "main", chart_type: str = "histogram", 
                                 columns: Optional[List[str]] = None, title: str = "Data Visualization") -> Dict[str, Any]:
        """Create a visualization"""
        return self._call_mcp_function(
            "create_visualization",
            dataset_name=dataset_name,
            chart_type=chart_type,
            columns=columns or [],
            title=title
        )
    
    async def train_model(self, dataset_name: str = "main", target_column: str = "", 
                         features: Optional[List[str]] = None, model_type: str = "linear_regression") -> Dict[str, Any]:
        """Train a machine learning model"""
        return self._call_mcp_function(
            "train_ml_model",
            dataset_name=dataset_name,
            target_column=target_column,
            features=features or [],
            model_type=model_type
        )
    
    async def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about datasets"""
        return self._call_mcp_function("get_dataset_info")
    
    async def describe_dataset(self, dataset_name: str = "main") -> Dict[str, Any]:
        """Get descriptive statistics for a dataset"""
        return self._call_mcp_function(
            "describe_data",
            dataset_name=dataset_name
        )
    
    async def correlation_analysis(self, dataset_name: str = "main", 
                                 columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform correlation analysis"""
        return self._call_mcp_function(
            "correlation_analysis",
            dataset_name=dataset_name,
            columns=columns or []
        )
    
    async def hypothesis_testing(self, dataset_name: str = "main", test_type: str = "ttest", 
                               columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Perform hypothesis testing"""
        test_kwargs = {"dataset_name": dataset_name, "test_type": test_type}
        if columns:
            test_kwargs["columns"] = columns
        test_kwargs.update(kwargs)
        
        return self._call_mcp_function("statistical_test", **test_kwargs)

# Create a global instance
mcp_client = MCPClient()
