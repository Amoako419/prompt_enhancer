"""Very simple test."""
print("Starting test...")
try:
    print("Testing imports...")
    import pandas as pd
    print("✅ Pandas imported")
    
    import numpy as np
    print("✅ NumPy imported")
    
    from mcp.server.fastmcp import FastMCP
    print("✅ FastMCP imported")
    
    print("Creating simple server...")
    test_server = FastMCP("Test")
    print("✅ Server created successfully")
    
    @test_server.tool()
    def simple_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    print("✅ Tool decorated successfully")
    print("Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
