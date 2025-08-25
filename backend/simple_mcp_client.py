"""
Simple MCP Client for Direct Communication

This module provides a simplified interface for communicating directly with the MCP server
without going through the standard MCP protocol. It imports the MCP server functions
directly and calls them as regular Python functions.
"""

import asyncio
import sys
import os
from pathlib import Path
import base64
import io
from typing import Any, Dict, List, Optional, Union

# Add MCP directory to path
mcp_dir = Path(__file__).parent.parent / "MCP"
sys.path.insert(0, str(mcp_dir))

# Import MCP server functions directly
try:
    # Import from the MCP main module
    import sys
    import os
    from pathlib import Path
    
    # Add MCP directory to Python path
    mcp_dir = Path(__file__).parent.parent / "MCP"
    if str(mcp_dir) not in sys.path:
        sys.path.insert(0, str(mcp_dir))
    
    # Now import the MCP functions
    from main import (
        mcp,
        load_data,
        load_csv_from_content,
        get_dataset_info,
        descriptive_statistics,
        correlation_analysis,
        data_visualization,
        statistical_test,
        train_ml_model,
        suggest_chart_types
    )
    
    # Try to import Image type - this might fail but that's okay
    try:
        from mcp.server.fastmcp.utilities.types import Image
    except ImportError:
        # Define a fallback Image class
        class Image:
            def __init__(self, data, media_type="image/png"):
                self.data = data
                self.media_type = media_type
    
    AVAILABLE = True
    print("MCP server functions imported successfully")
    
except ImportError as e:
    print(f"Warning: Could not import MCP server functions: {e}")
    mcp = None
    AVAILABLE = False

class SimpleMCPClient:
    """Simplified MCP client that calls server functions directly"""
    
    def __init__(self):
        self.available = AVAILABLE
        
    def _serialize_data(self, data: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        if isinstance(data, Image):
            # Handle Image objects by extracting the base64 data
            if hasattr(data, 'data'):
                return {
                    'type': 'image',
                    'data': data.data,
                    'format': getattr(data, 'format', 'png')
                }
            else:
                return {'type': 'image', 'data': str(data)}
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        else:
            return data
    
    async def load_data(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load data from file"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await load_data(file_path=file_path, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def load_csv_from_content(self, content: str, filename: str, dataset_name: str = "main", **kwargs) -> Dict[str, Any]:
        """Load CSV data from string content"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await load_csv_from_content(content=content, filename=filename, dataset_name=dataset_name, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await get_dataset_info(dataset_name=dataset_name)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def descriptive_statistics(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate descriptive statistics"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await descriptive_statistics(dataset_name=dataset_name, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def correlation_analysis(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Perform correlation analysis"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await correlation_analysis(dataset_name=dataset_name, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def data_visualization(self, dataset_name: str, chart_type: str, **kwargs) -> Dict[str, Any]:
        """Generate data visualizations"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await data_visualization(dataset_name=dataset_name, chart_type=chart_type, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def statistical_test(self, dataset_name: str, test_type: str, **kwargs) -> Dict[str, Any]:
        """Perform statistical tests"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await statistical_test(dataset_name=dataset_name, test_type=test_type, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def train_ml_model(self, dataset_name: str, model_type: str, **kwargs) -> Dict[str, Any]:
        """Train machine learning models"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await train_ml_model(dataset_name=dataset_name, model_type=model_type, **kwargs)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def get_chart_recommendations(self, dataset_name: str) -> Dict[str, Any]:
        """Get intelligent chart recommendations based on data characteristics"""
        if not self.available:
            return {"error": "MCP server not available"}
        
        try:
            result = await suggest_chart_types(dataset_name=dataset_name)
            return self._serialize_data(result)
        except Exception as e:
            return {"error": str(e)}

# Create a singleton instance
simple_mcp_client = SimpleMCPClient()
