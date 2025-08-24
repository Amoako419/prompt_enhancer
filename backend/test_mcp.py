import sys
sys.path.append('.')
from mcp_client import mcp_client
print('MCP Available:', mcp_client.available)

if mcp_client.available:
    print('✅ MCP client loaded successfully')
else:
    print('❌ MCP client failed to load')
