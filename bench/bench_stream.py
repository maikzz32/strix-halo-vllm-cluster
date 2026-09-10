"""Reproducible OpenAI-compatible SSE benchmark; saves full outputs and timing."""
import argparse, concurrent.futures, datetime, json, pathlib, time, urllib.request

PROMPTS = {
    'code': 'Write a complete Python implementation of a bounded LRU cache with time-to-live expiry, a thread-safe API, and unit tests covering expiry, eviction, updates, and concurrent reads. Explain the design and its complexity. Use only the standard library.',
    'german': 'Erklaere ausfuehrlich, wie ein verteilter Datenbankdienst mit vier Servern Schreibzugriffe, Replikation und Netzwerkausfaelle behandelt. Vergleiche konkrete Alternativen anhand eines Warenlagers und zeige die Folgen fuer Konsistenz und Verfuegbarkeit. Fuehre deine Erklaerung mit Beispielen und einer Entscheidungsmatrix fort.',
    'analysis': 'Design a reproducible experiment to distinguish a memory bandwidth bottleneck from a network latency bottleneck in a four-node language model inference cluster. Derive the expected scaling behavior, specify measurements and confounders, and explain how to interpret contradictory observations in detail.',
}

def request_json(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)

def sample(args, model, name, repeat):
    body = {'model':model, 'messages':[{'role':'user','content':PROMPTS[name]}],
            'max_tokens':args.tokens, 'temperature':0, 'seed':42, 'stream':True,
            'stream_options':{'include_usage':True,'continuous_usage_stats':True},
            'chat_template_kwargs':{'enable_thinking':False}, 'ignore_eos':args.ignore_eos}
    req = urllib.request.Request(args.url+'/v1/chat/completions', json.dumps(body).encode(), {'Content-Type':'application/json'})
    t0 = time.perf_counter()
    first = last = None
    first_count = 0
    events = []
    content = []
    reasoning = []
    usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            obj = json.loads(data)
            now = time.perf_counter()
            if obj.get('error'):
                raise RuntimeError(obj['error'])
            if obj.get('usage'):
                usage = obj['usage']
            choices = obj.get('choices', [])
            if not choices:
                continue
            c = choices[0]
            delta = c.get('delta', {})
            text = delta.get('content') or ''
            think = delta.get('reasoning') or delta.get('reasoning_content') or ''
            if text or think:
                count = usage.get('completion_tokens')
                if first is None:
                    first = now
                    first_count = count or 1
                last = now
                events.append({'t':now-t0,'completion_tokens':count,'chars':len(text)+len(think)})
                content.append(text)
                reasoning.append(think)
            finish = c.get('finish_reason') or finish
    end = time.perf_counter()
    n = usage.get('completion_tokens', 0)
    decode_seconds = last-first if first is not None and last is not None else 0
    result = {'prompt':name,'repeat':repeat,'request':body,'usage':usage,'ttft_s':None if first is None else first-t0,
              'wall_s':end-t0,'decode_s':decode_seconds,'first_chunk_tokens':first_count,
              'decode_tps':(n-first_count)/decode_seconds if decode_seconds>0 else None,
              'end_to_end_tps':n/(end-t0),'finish_reason':finish,'events':events,
              'content':''.join(content),'reasoning':''.join(reasoning)}
    print(json.dumps({k:result[k] for k in ['prompt','repeat','usage','ttft_s','wall_s','decode_tps','end_to_end_tps','finish_reason']}, ensure_ascii=False), flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--output',required=True)
    ap.add_argument('--tokens',type=int,default=512)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--concurrency',type=int,default=1)
    ap.add_argument('--prompts',default=','.join(PROMPTS))
    ap.add_argument('--ignore-eos',action='store_true')
    args=ap.parse_args()
    models=request_json(args.url+'/v1/models')
    model=models['data'][0]['id']
    jobs=[(n,r) for r in range(args.repeats) for n in args.prompts.split(',')]
    started=time.perf_counter()
    results=[]
    target=pathlib.Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(sample,args,model,n,r) for n,r in jobs]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'arguments':vars(args), 'models':models,'results':results,'elapsed_s':time.perf_counter()-started}
            target.write_text(json.dumps(report,indent=2,ensure_ascii=False), encoding='utf-8')
    print('SAVED',target,flush=True)

if __name__=='__main__':
    main()
