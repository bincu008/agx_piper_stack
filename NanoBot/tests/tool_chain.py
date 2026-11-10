"""Pure synthetic arithmetic tools: no robot calls, network tools, or ROS substitutes."""
import json,time,pathlib
from openai import OpenAI
c=OpenAI(base_url='http://127.0.0.1:1234/v1',api_key='local',timeout=90,max_retries=0)
tool={'type':'function','function':{'name':'calculate','description':'Synthetic arithmetic test. Returns sum of two integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}
m=[{'role':'system','content':'Follow the test exactly. Use structured tool calls. Call calculate with 7 and 5. Then use its result plus 9 in another call. Then use that result plus 4. Finally report the final result. Never calculate without the tool.'},{'role':'user','content':'Run the three-step arithmetic test.'}]; log=[]
try:
 for step in range(5):
  r=c.chat.completions.create(model='robobrain2.5-4b',messages=m,tools=[tool],temperature=0,max_tokens=768)
  x=r.choices[0].message; m.append(x.model_dump(exclude_none=True)); log.append(x.model_dump(exclude_none=True))
  if not x.tool_calls: break
  for t in x.tool_calls:
   a=json.loads(t.function.arguments); assert t.function.name=='calculate' and set(a)=={'a','b'}
   value=a['a']+a['b']; m.append({'role':'tool','tool_call_id':t.id,'content':json.dumps({'result':value})})
 calls=[t for x in log for t in x.get('tool_calls',[])]; passed=len(calls)==3 and '25' in (log[-1].get('content') or '')
 result={'passed':passed,'calls':len(calls),'transcript':m}
except Exception as e: result={'passed':False,'error':str(e),'transcript':m}
pathlib.Path('results/tool-chain.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
