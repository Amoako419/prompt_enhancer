"""Quick test to verify the server can load."""
try:
    from main import mcp
    print("✅ Server loaded successfully!")
    print("\n📋 Available tools:")
    tools = mcp._tool_manager.list_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    print(f"\n📊 Total tools: {len(tools)}")
    print("\n🎯 Server name:", mcp.name)
    print("📝 Instructions:", mcp.instructions[:100] + "..." if mcp.instructions else "None")
except Exception as e:
    print(f"❌ Error loading server: {e}")
    import traceback
    traceback.print_exc()
