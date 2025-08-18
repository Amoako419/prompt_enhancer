"""Test the main server loading."""
print("Testing main server...")
try:
    from main import mcp
    print("✅ Main server imported successfully!")
    
    tools = mcp._tool_manager.list_tools()
    print(f"✅ Found {len(tools)} tools:")
    for tool in tools[:5]:  # Show first 5 tools
        print(f"  - {tool.name}")
    
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more tools")
    
    print(f"✅ Server name: {mcp.name}")
    print("✅ Server loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading main server: {e}")
    import traceback
    traceback.print_exc()
