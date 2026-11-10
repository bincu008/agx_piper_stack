"""Native Nanobot provider verification; only local synthetic arithmetic executes."""
import asyncio,json,pathlib,dataclasses,time
from nanobot.config.loader import load_config
from nanobot.providers.factory import make_provider
async def main():
 c=load_config();p=make_provider(c);d=c.agents.defaults;report={};kw=dict(model=d.model,max_tokens=d.max_tokens,temperature=0,reasoning_effort=d.reasoning_effort)
 start=time.monotonic()
 r=await p.chat(messages=[{'role':'user','content':'What is 1 + 1? Answer with the number only.'}],**kw)
 report['text']={'content':r.content,'finish_reason':r.finish_reason};assert '2' in (r.content or ''),report
 tool={'type':'function','function':{'name':'synthetic_add','description':'Synthetic test only. Add two integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}
 m=[{'role':'system','content':'Use synthetic_add to add 17 and 25. Wait for the actual tool result, then report it as RESULT=<number>. Never invent tool results.'},{'role':'user','content':'Run the synthetic addition test.'}]
 r=await p.chat(messages=m,tools=[tool],tool_choice='required',**kw);assert len(r.tool_calls)==1,repr(r)
 t=r.tool_calls[0];assert t.name=='synthetic_add' and t.arguments=={'a':17,'b':25},repr(t)
 m.append({'role':'assistant','content':r.content or '', 'tool_calls':[{'id':t.id,'type':'function','function':{'name':t.name,'arguments':json.dumps(t.arguments)}}]});m.append({'role':'tool','tool_call_id':t.id,'content':json.dumps({'result':42})})
 final=await p.chat(messages=m,tools=[tool],tool_choice='none',**kw)
 assert '42' in (final.content or '') and not final.tool_calls,repr(final)
 report['tool_roundtrip']={'name':t.name,'arguments':t.arguments,'local_result':42,'final':final.content,'passed':True};report['seconds']=round(time.monotonic()-start,2);report['robot_commands_executed']=0
 pathlib.Path('results/remote-qwen-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
asyncio.run(main())
