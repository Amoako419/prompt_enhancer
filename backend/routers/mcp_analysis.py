from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from simple_mcp_client import simple_mcp_client as mcp_client
from typing import Optional, Dict, Any, Union
import json
import numpy as np
import pandas as pd

router = APIRouter(tags=["MCP Data Analysis"])

def serialize_for_json(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return serialize_for_json(obj.__dict__)
    else:
        return str(obj) if not isinstance(obj, (int, float, str, bool, type(None))) else obj

@router.get("/health")
async def health_check():
    """Health check for MCP services"""
    return {"status": "healthy", "service": "mcp-data-analysis"}

@router.get("/debug/datasets")
async def debug_datasets():
    """Debug endpoint to check MCP server availability"""
    if not mcp_client.available:
        return {
            "status": "error",
            "message": "MCP server not available",
            "mcp_available": False
        }
    
    try:
        # Try to get info about the main dataset
        dataset_info = await mcp_client.get_dataset_info("main")
        if "error" in dataset_info:
            return {
                "status": "no_data",
                "message": "No datasets loaded in MCP server",
                "mcp_available": True,
                "datasets": {}
            }
        
        return {
            "status": "success",
            "mcp_available": True,
            "main_dataset": dataset_info
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking MCP datasets: {str(e)}",
            "mcp_available": True
        }

@router.post("/load-data")
async def load_data(file: UploadFile = File(...)):
    """Load data from uploaded file using MCP server"""
    try:
        # Read file content with proper encoding handling
        content = await file.read()
        
        # Try different encodings to handle various file formats
        file_content = None
        for encoding in ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']:
            try:
                file_content = content.decode(encoding)
                # Remove BOM if present
                if file_content.startswith('\ufeff'):
                    file_content = file_content[1:]
                break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            return {
                "status": "error",
                "message": "Could not decode file. Please ensure it's a valid text file with proper encoding.",
                "results": {}
            }
        
        # Use MCP client to load CSV data
        result = await mcp_client.load_csv_from_content(
            content=file_content,
            filename=file.filename or "uploaded_data.csv",
            dataset_name="main"
        )
        
        print(f"DEBUG: MCP result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        print(f"DEBUG: MCP result error check: {'error' in result if isinstance(result, dict) else 'N/A'}")
        
        if isinstance(result, dict) and "error" in result:
            response = {
                "status": "error",
                "message": result["error"],
                "results": {}
            }
        else:
            # Serialize the result for JSON response
            serialized_result = serialize_for_json(result)
            print(f"DEBUG: Serialized result keys: {list(serialized_result.keys()) if isinstance(serialized_result, dict) else 'not a dict'}")
            response = {
                "status": "success",
                "message": f"Data loaded successfully from {file.filename}",
                "results": serialized_result
            }
        
        # Apply serialization to the entire response to be safe
        return serialize_for_json(response)
        
    except Exception as e:
        # Return example data if file loading fails
        try:
            example_data = '''name,age,salary,city
John,25,50000,New York
Jane,30,60000,Los Angeles
Bob,35,70000,Chicago
Alice,28,55000,Miami
Charlie,32,65000,Seattle'''
            result = await mcp_client.load_csv_from_content(example_data, "example.csv", "main")
            serialized_result = serialize_for_json(result)
            response = {
                "status": "success",
                "message": "Using example data due to file processing error",
                "results": serialized_result,
                "note": f"Error: {str(e)}"
            }
            return serialize_for_json(response)
        except Exception:
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/descriptive-stats")
async def descriptive_stats(file: Optional[UploadFile] = File(None)):
    """Get descriptive statistics using MCP server"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {"error": load_result["error"]}
                }
        
        # Get descriptive statistics using MCP client
        stats_result = await mcp_client.describe_dataset("main")
        
        if "error" in stats_result:
            return {
                "status": "error",
                "message": stats_result["error"],
                "results": {
                    "title": "No Data Available",
                    "content": "Please upload a CSV file first to perform analysis.",
                    "error": stats_result["error"]
                }
            }
        
        # Serialize the result for JSON response
        serialized_result = serialize_for_json(stats_result)
        
        # Format the results for the frontend
        return {
            "status": "success",
            "message": "Descriptive statistics completed",
            "results": {
                "title": "Descriptive Statistics Analysis",
                "content": "Statistical analysis completed using MCP server.",
                "data": serialized_result
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error processing your data: {str(e)}",
            "results": {
                "title": "Analysis Error",
                "content": f"Failed to analyze your uploaded data: {str(e)}\n\nPlease check your file format and try again.",
                "data": {"error": str(e)}
            }
        }

@router.post("/correlation-analysis")
async def correlation_analysis(
    file: Optional[UploadFile] = File(None),
    method: Optional[str] = Form("pearson")
):
    """Perform correlation analysis using MCP server"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {"error": load_result["error"]}
                }
        
        # Perform correlation analysis using MCP client
        corr_result = await mcp_client.correlation_analysis("main", method)
        
        if "error" in corr_result:
            return {
                "status": "error",
                "message": corr_result["error"],
                "results": {
                    "title": "No Data Available",
                    "content": "Please upload a CSV file first to perform correlation analysis.",
                    "error": corr_result["error"]
                }
            }
        
        # Format the results for the frontend
        return {
            "status": "success",
            "message": "Correlation analysis completed",
            "results": {
                "title": "Correlation Analysis Results",
                "content": f"Correlation analysis completed using {method} method with MCP server.",
                "data": corr_result
            }
        }
        print(f"DEBUG: Got correlation analysis for {data_id}")
        
        # Format the results with REAL analysis insights
        strong_correlations = result.get('strong_correlations', [])
        variables_analyzed = result.get('variables_analyzed', [])
        correlation_summary = f"Found {len(strong_correlations)} strong correlations" if strong_correlations else "No strong correlations detected"
        
        correlation_insights = []
        if strong_correlations:
            for corr in strong_correlations[:3]:
                correlation_insights.append(f"• {corr['variable1']} ↔ {corr['variable2']}: r = {corr['correlation']:.3f} ({corr['strength']})")
        else:
            correlation_insights.append("• All correlations are weak (|r| < 0.7)")
        
        formatted_results = {
            "title": "Correlation Analysis Results",
            "content": f"Correlation analysis completed using {method} method on your uploaded data.\n\n📊 Analysis Summary:\n{correlation_summary}\n\n🔍 Variables Analyzed: {len(variables_analyzed)}\n{', '.join(variables_analyzed)}\n\n🎯 Strong Correlations (|r| > 0.7):\n" + 
                      "\n".join(correlation_insights),
            "data": result
        }
        
        return {
            "status": "success",
            "message": "Correlation analysis completed on your data",
            "results": formatted_results
        }
    except Exception as e:
        print(f"DEBUG: Error in correlation_analysis: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing correlation analysis: {str(e)}",
            "results": {
                "title": "Analysis Error",
                "content": f"Failed to perform correlation analysis: {str(e)}",
                "data": {"error": str(e)}
            }
        }

@router.post("/chart-recommendations")
async def get_chart_recommendations(
    file: Optional[UploadFile] = File(None),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None)
):
    """Get intelligent chart recommendations based on data characteristics"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {"error": load_result["error"]}
                }
        
        # Get dataset info to help with column selection
        dataset_info = await mcp_client.describe_dataset("main")
        if "error" in dataset_info:
            return {
                "status": "error", 
                "message": "No data available",
                "results": {"error": "Please upload data first"}
            }
        
        # Auto-select columns if not provided
        if not x_column:
            numeric_cols = [col for col, dtype in dataset_info.get("dtypes", {}).items() 
                          if "int" in str(dtype) or "float" in str(dtype)]
            if numeric_cols:
                x_column = numeric_cols[0]
            else:
                x_column = list(dataset_info.get("columns", {}))[0] if dataset_info.get("columns") else ""
        
        # Get chart recommendations
        recommendations = await mcp_client.get_chart_recommendations(
            dataset_name="main",
            x_column=x_column,
            y_column=y_column
        )
        
        if "error" in recommendations:
            return {
                "status": "error",
                "message": recommendations["error"],
                "results": {"error": recommendations["error"]}
            }
        
        return {
            "status": "success",
            "message": "Chart recommendations generated successfully",
            "results": {
                "title": "Smart Chart Recommendations",
                "content": f"Analyzed data characteristics for {x_column}" + (f" and {y_column}" if y_column else ""),
                "chart_recommendations": recommendations.get("recommendations", []),
                "data_insights": recommendations.get("data_insights", {}),
                "selected_columns": {
                    "x_column": x_column,
                    "y_column": y_column
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating chart recommendations: {str(e)}",
            "results": {
                "title": "Recommendation Error",
                "content": f"Failed to generate chart recommendations: {str(e)}",
                "error": str(e)
            }
        }

@router.post("/visualization")
async def create_visualization(
    file: Optional[UploadFile] = File(None),
    chart_type: Optional[str] = Form("histogram"),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None)
):
    """Generate visualization using MCP server"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {"error": load_result["error"]}
                }
        
        # Auto-select columns if not provided
        if not x_column:
            dataset_info = await mcp_client.get_dataset_info("main")
            if "error" not in dataset_info and "columns" in dataset_info:
                numeric_cols = [col for col, info in dataset_info["columns"].items() 
                              if info.get("dtype") in ["int64", "float64"]]
                x_column = numeric_cols[0] if numeric_cols else list(dataset_info["columns"].keys())[0]
        
        # Create visualization using MCP server
        viz_result = await mcp_client.data_visualization(
            dataset_name="main",
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            title=f"{chart_type.title()} of {x_column}"
        )
        
        if "error" in viz_result:
            return {
                "status": "error",
                "message": viz_result["error"],
                "results": {"error": viz_result["error"]}
            }
        
        # Format response for frontend
        # Handle the fact that MCP returns a single Image object
        visualizations = []
        if viz_result and "error" not in viz_result:
            # If viz_result is a single image object (serialized by simple_mcp_client)
            if isinstance(viz_result, dict) and viz_result.get("type") == "image":
                visualizations = [viz_result]
            # If viz_result has a visualizations array
            elif "visualizations" in viz_result:
                visualizations = viz_result["visualizations"]
            # If viz_result is the image data itself
            else:
                visualizations = [viz_result]
       
        return {
            "status": "success",
            "message": "Visualization created successfully",
            "results": {
                "title": f"Data Visualization: {chart_type.title()}",
                "content": f"Created {chart_type} visualization using MCP server.",
                "visualizations": visualizations
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating visualization: {str(e)}",
            "results": {
                "title": "Visualization Error",
                "content": f"Failed to create visualization: {str(e)}",
                "error": str(e)
            }
        }
        return {
            "status": "error",
            "message": f"Error creating visualization: {str(e)}",
            "results": {
                "title": "Visualization Error",
                "content": f"Failed to create visualization: {str(e)}",
                "data": {"error": str(e)}
            }
        }

@router.post("/smart-visualization")
async def create_smart_visualization(
    file: Optional[UploadFile] = File(None),
    x_column: Optional[str] = Form(""),
    y_column: Optional[str] = Form(""),
    color_column: Optional[str] = Form("")
):
    """Generate intelligent visualizations based on data characteristics using MCP server"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {"error": load_result["error"]}
                }
        
        # Get dataset info to validate columns
        dataset_info = await mcp_client.describe_dataset("main")
        if "error" in dataset_info:
            return {
                "status": "error",
                "message": "No data available for visualization",
                "results": {"error": "No dataset loaded"}
            }
        
        # Auto-select columns if not provided
        if not x_column:
            # Use first numeric column if available, otherwise first column
            numeric_cols = [col for col in dataset_info.get("columns", {}).keys() 
                          if dataset_info.get("dtypes", {}).get(col, "").startswith(('int', 'float'))]
            if numeric_cols:
                x_column = numeric_cols[0]
            else:
                x_column = list(dataset_info.get("columns", {}).keys())[0] if dataset_info.get("columns") else ""
        
        # Create smart visualization using MCP server
        viz_result = await mcp_client.create_smart_visualization(
            dataset_name="main",
            x_column=x_column,
            y_column=y_column if y_column else None,
            color_column=color_column if color_column else None
        )
        
        if "error" in viz_result:
            return {
                "status": "error",
                "message": viz_result["error"],
                "results": {"error": viz_result["error"]}
            }
        
        # Format response for frontend
        return {
            "status": "success",
            "message": "Smart visualizations created successfully",
            "results": {
                "title": "Intelligent Data Visualizations",
                "content": f"Created {viz_result.get('suggestions_count', 0)} intelligent visualizations based on data characteristics.",
                "visualizations": viz_result.get("visualizations", []),
                "data_info": viz_result.get("data_info", {}),
                "x_column": x_column,
                "y_column": y_column,
                "suggestions_count": viz_result.get("suggestions_count", 0)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating smart visualization: {str(e)}",
            "results": {
                "title": "Smart Visualization Error",
                "content": f"Failed to create intelligent visualizations: {str(e)}",
                "error": str(e)
            }
        }

@router.post("/hypothesis-testing")
async def hypothesis_testing(
    file: Optional[UploadFile] = File(None),
    test_type: Optional[str] = Form("t_test"),
    alpha: Optional[float] = Form(0.05)
):
    """Perform hypothesis testing using MCP server"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {
                        "title": "Data Loading Error",
                        "content": f"Failed to load data: {load_result['error']}",
                        "data": {"error": load_result["error"]}
                    }
                }
        
        # Perform statistical test using MCP client
        test_result = await mcp_client.statistical_test(
            dataset_name="main",
            test_type=test_type,
            alpha=alpha
        )
        
        if "error" in test_result:
            return {
                "status": "error",
                "message": test_result["error"],
                "results": {
                    "title": "No Data Available",
                    "content": "Please upload a CSV file first to perform hypothesis testing.",
                    "data": {"error": test_result["error"]}
                }
            }
        
        # Format the results for the frontend
        return {
            "status": "success",
            "message": "Hypothesis testing completed",
            "results": {
                "title": "Hypothesis Testing Results",
                "content": f"Statistical test ({test_type}) completed using MCP server.",
                "data": test_result
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing hypothesis testing: {str(e)}",
            "results": {
                "title": "Hypothesis Testing Error",
                "content": f"Failed to perform hypothesis testing: {str(e)}",
                "data": {"error": str(e)}
            }
        }

@router.post("/machine-learning")
async def machine_learning(
    file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
    model_type: Optional[str] = Form("kmeans"),
    n_clusters: Optional[int] = Form(3),
    features: Optional[str] = Form(None)
):
    """Perform machine learning analysis using MCP server"""
    try:
        # If we have file data, load it first
        if file and file.filename:
            content = await file.read()
            file_content = content.decode('utf-8')
            load_result = await mcp_client.load_csv_from_content(
                content=file_content,
                filename=file.filename,
                dataset_name="main"
            )
            if "error" in load_result:
                return {
                    "status": "error",
                    "message": load_result["error"],
                    "results": {"error": load_result["error"]}
                }
        
        # Parse features if provided
        feature_list = None
        if features:
            try:
                feature_list = json.loads(features) if features.startswith('[') else features.split(',')
                feature_list = [f.strip() for f in feature_list]
            except:
                feature_list = None
        
        # For clustering (default), set up appropriate parameters
        if model_type == "kmeans" or not target_column:
            # Use train_ml_model with kmeans
            ml_result = await mcp_client.train_ml_model(
                dataset_name="main",
                target_column="",  # Empty for clustering
                model_type="kmeans",
                features=feature_list
            )
        else:
            # For supervised learning
            ml_result = await mcp_client.train_ml_model(
                dataset_name="main",
                target_column=target_column,
                model_type=model_type,
                features=feature_list
            )
        
        if "error" in ml_result:
            return {
                "status": "error",
                "message": ml_result["error"],
                "results": {"error": ml_result["error"]}
            }
        
        # Format response for frontend
        return {
            "status": "success",
            "message": "Machine learning analysis completed",
            "results": {
                "title": "Machine Learning Analysis Results",
                "content": f"Completed {model_type} analysis using MCP server.",
                "model_results": ml_result,
                "visualizations": [ml_result.get("plot")] if ml_result.get("plot") else []
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in machine learning analysis: {str(e)}",
            "results": {
                "title": "Machine Learning Error",
                "content": f"Failed to perform machine learning analysis: {str(e)}",
                "error": str(e)
            }
        }
        return {
            "status": "error",
            "message": "No dataset available for clustering analysis",
            "results": {
                "title": "Clustering Analysis (No Data)",
                "content": "No dataset found. Please upload data first.",
                "data": {"error": "No dataset available"}
            }
        }
    except Exception as e:
        print(f"DEBUG: Error in machine_learning (clustering): {str(e)}")
        return {
            "status": "error",
            "message": f"Error in clustering analysis: {str(e)}",
            "results": {
                "title": "Clustering Analysis Error",
                "content": f"Failed to perform clustering analysis: {str(e)}",
                "data": {"error": str(e)}
            }
        }

@router.post("/chart-recommendations")
async def get_chart_recommendations(dataset_name: str = "main"):
    """Get intelligent chart recommendations based on dataset characteristics"""
    try:
        if not mcp_client.available:
            return {
                "status": "error",
                "message": "MCP server not available",
                "recommendations": []
            }
        
        result = await mcp_client.get_chart_recommendations(dataset_name)
        
        if "error" in result:
            return {
                "status": "error",
                "message": result["error"],
                "recommendations": []
            }
        
        return {
            "status": "success",
            "message": "Chart recommendations generated successfully",
            "dataset_info": result.get("dataset_info", {}),
            "recommendations": result.get("recommendations", []),
            "analysis_summary": result.get("analysis_summary", ""),
            "total_recommendations": result.get("total_recommendations", 0)
        }
        
    except Exception as e:
        print(f"DEBUG: Error in chart recommendations: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting chart recommendations: {str(e)}",
            "recommendations": []
        }
