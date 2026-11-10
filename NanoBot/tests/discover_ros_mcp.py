import asyncio,json,pathlib,httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
root=pathlib.Path(__file__).resolve().parents[1]
async def main():
 token=(root/'.private/ros-mcp-token').read_text().strip()
 async with httpx.AsyncClient(headers={'Authorization':'Bearer '+token},timeout=30) as h:
  async with streamable_http_client('https://10.211.55.3:8765/mcp',http_client=h) as (r,w,_):
   async with ClientSession(r,w) as s:
    init=await s.initialize(); tools=[];cursor=None
    while True:
     page=await s.list_tools(cursor=cursor);tools.extend(page.tools);cursor=page.nextCursor
     if not cursor:break
    (root/'results/mcp-tools.json').write_text(json.dumps([t.model_dump(mode='json') for t in tools],indent=2))
    print('Server',init.serverInfo,'tools',len(tools))
    for t in tools:print(t.name,(t.description or '')[:160],json.dumps(t.inputSchema)[:250])
    for name in ['list_functions','get_arm_state','get_world_state','get_spatial_context']:
     result=await s.call_tool(name,{})
     (root/'results'/('mcp-status-'+name+'.json')).write_text(result.model_dump_json(indent=2))
     data=result.structuredContent or {}
     print('READ',name,'isError',result.isError,'mode',data.get('mode'),'transport',data.get('transport_outcome'),'status',(data.get('response') or {}).get('status'))
    resources=await s.list_resources();print('Resources',[r.uri for r in resources.resources])
    for uri in ['ros-pipeline://catalog','ros-pipeline://workflow','ros-pipeline://agent-pipeline','ros-pipeline://spatial-map','ros-pipeline://nanobot']:
     data=await s.read_resource(uri);(root/'results'/('mcp-resource-'+uri.split('://')[1]+'.json')).write_text(data.model_dump_json(indent=2))
asyncio.run(asyncio.wait_for(main(),60))
