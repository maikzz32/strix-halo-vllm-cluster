"""Check auto tool selection, streaming, and a synthetic tool-result round trip."""
import json,urllib.request,pathlib,argparse
URL='http://192.168.1.15:8000/v1/chat/completions'
MODEL='/home/cluster-user/qwen38_rest'
TOOL={'type':'function','function':{'name':'get_weather','description':'Get the current temperature for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}
def request(body):
 return urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output",type=pathlib.Path,default=pathlib.Path("results/qwen029-tools-20260909.json"));args=parser.parse_args()
 report={}
 base={'model':MODEL,'messages':[{'role':'user','content':'Use get_weather to get the current weather in Berlin.'}],'tools':[TOOL],'tool_choice':'auto','temperature':0,'max_tokens':512,'chat_template_kwargs':{'enable_thinking':False}}
 with request(base) as r:answer=json.load(r)
 msg=answer['choices'][0]['message'];tc=msg['tool_calls'];assert tc[0]['function']['name']=='get_weather';assert json.loads(tc[0]['function']['arguments'])['city'].lower()=='berlin';report['nonstream']=answer
 follow=dict(base);follow['messages']=base['messages']+[msg,{'role':'tool','tool_call_id':tc[0]['id'],'content':'{"city":"Berlin","temperature_c":21}'}];follow['tool_choice']='none'
 with request(follow) as r:answer=json.load(r)
 assert '21' in answer['choices'][0]['message']['content'];report['roundtrip']=answer
 calls={};events=[]
 with request(dict(base,stream=True)) as response:
  for raw in response:
   line=raw.decode().strip()
   if not line.startswith('data: ') or line=='data: [DONE]':continue
   e=json.loads(line[6:]);events.append(e)
   for ch in e.get('choices',[]):
    for t in ch.get('delta',{}).get('tool_calls',[]):
     c=calls.setdefault(t['index'],{'name':'','arguments':''});f=t.get('function',{});c['name']+=f.get('name') or '';c['arguments']+=f.get('arguments') or ''
 assert calls[0]['name']=='get_weather';assert json.loads(calls[0]['arguments'])['city'].lower()=='berlin';report['stream_calls']=calls
 default=dict(base);default.pop('chat_template_kwargs');default['max_tokens']=1024
 with request(default) as r:answer=json.load(r)
 tc=answer['choices'][0]['message']['tool_calls'];assert tc[0]['function']['name']=='get_weather';assert json.loads(tc[0]['function']['arguments'])['city'].lower()=='berlin';report['default_thinking']=answer
 p=args.output;p.write_text(json.dumps(report,indent=2),encoding='utf-8');print('PASS: auto tool selection, result roundtrip, streamed tool call, default thinking')
if __name__=='__main__':main()
