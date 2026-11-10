"""Read-only MCP discovery. Never invokes a server tool; registers none by default."""
import argparse, asyncio, ipaddress, json, os, pathlib, shutil
from urllib.parse import urlparse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
p=argparse.ArgumentParser();p.add_argument('url');a=p.parse_args();u=urlparse(a.url)
assert u.scheme in ('http','https') and u.hostname=='10.211.55.3', 'Only discovered VM IP is permitted'
token=os.environ.get('ROBOT_MCP_TOKEN');assert token, 'Set ROBOT_MCP_TOKEN using the existing server credential'
async def main():
 async with streamablehttp_client(a.url,headers={'Authorization':'Bearer '+token}) as (r,w,_):
  async with ClientSession(r,w) as s:
   await s.initialize(); tools=await s.list_tools();pathlib.Path('results/mcp-tools.json').write_text(tools.model_dump_json(indent=2))
 # Native remote MCP; expose nothing until read-only tool names have been reviewed.
 path=pathlib.Path.home()/'.nanobot/config.json';c=json.loads(path.read_text());shutil.copy2(path,str(path)+'.before-mcp')
 c['tools']['ssrfWhitelist']=['10.211.55.3/32'];c['tools']['mcpServers']['ros2']={'type':'streamableHttp','url':a.url,'headers':{'Authorization':'Bearer '+token},'toolTimeout':30,'enabledTools':[]}
 path.write_text(json.dumps(c,indent=2)+'\n');path.chmod(0o600); print('Discovered schemas saved. Native MCP configured with zero callable tools pending review.')
asyncio.run(asyncio.wait_for(main(),30))
