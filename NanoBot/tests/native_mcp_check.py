import asyncio,json,pathlib
from nanobot.config.loader import load_config
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.security.network import configure_ssrf_whitelist
async def main():
 c=load_config();configure_ssrf_whitelist(c.tools.ssrf_whitelist);r=ToolRegistry();p=MCPProvider.from_config(c,r)
 try:
  await p.connect();print('Connected:',p.connected_server_names);print('Tools:',r.tool_names)
  assert len(r.tool_names)==len(c.tools.mcp_servers["ros2"].enabled_tools)
  assert not any('move_to' in n or 'execute_plan' in n for n in r.tool_names)
  result=await r.execute('mcp_ros2_get_arm_state',{})
  text=str(result);print('Read-only arm result:',text[:1200]);assert 'ARM_STATE_READY' in text
 finally:await p.aclose()
asyncio.run(asyncio.wait_for(main(),60))
