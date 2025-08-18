"""Minimal test of the server components."""
print("Testing core components...")

# Test matplotlib configuration
print("1. Testing matplotlib...")
try:
    import matplotlib
    matplotlib.use('Agg')
    print("✅ Matplotlib configured")
except Exception as e:
    print(f"❌ Matplotlib error: {e}")

# Test pandas/numpy
print("2. Testing data libraries...")
try:
    import pandas as pd
    import numpy as np
    print("✅ Data libraries OK")
except Exception as e:
    print(f"❌ Data libraries error: {e}")

# Test FastMCP
print("3. Testing FastMCP...")
try:
    from mcp import FastMCP
    print("✅ FastMCP import OK")
except Exception as e:
    print(f"❌ FastMCP error: {e}")

print("Component test completed!")
