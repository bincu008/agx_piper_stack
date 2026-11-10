import asyncio,json,pathlib,sys,time
from nanobot.utils.token_encoding import warmup_token_encoding,get_token_encoding
warmup_token_encoding()
while get_token_encoding() is None: time.sleep(0.05)
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import mcp_profiles
from nanobot.config.loader import load_config
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.security.network import configure_ssrf_whitelist
from nanobot.utils.helpers import estimate_prompt_tokens_chain
async def main():
 results=[]
 for name in mcp_profiles.P:
  c=load_config(mcp_profiles.generate(name));configure_ssrf_whitelist(c.tools.ssrf_whitelist);r=ToolRegistry();loop=AgentLoop.from_config(c,tool_registry=r);p=MCPProvider.from_config(c,r)
  try:
   await p.connect();registered=[n for n in r.tool_names if n.startswith('mcp_ros2_')];assert len(registered)==len(c.tools.mcp_servers['ros2'].enabled_tools)
   messages=loop.context.build_messages([], 'Check the current task prerequisites. Do not execute any tools.',channel='cli')
   tokens,source=estimate_prompt_tokens_chain(loop.provider,c.agents.defaults.model,messages,r.get_definitions())
   row={'profile':name,'tools':len(registered),'estimated_input_tokens':tokens,'estimator':source,'fits_initial_input_budget':tokens<30200};print(row);results.append(row)
   if name=='view':
    result=await r.execute('mcp_ros2_get_observe_view',{'dx':0,'dy':0,'dz':0,'roll':0,'pitch':0,'yaw':0});pathlib.Path('results/native-view-calculation.txt').write_text(str(result));print('VIEW CALC',str(result)[:500])
  finally:await p.aclose()
 pathlib.Path('results/mcp-profile-checks.json').write_text(json.dumps(results,indent=2))
asyncio.run(asyncio.wait_for(main(),90))
