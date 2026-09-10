"""Export vLLM results with output hashes, without publishing generated text."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

def export(result, config, run_id):
    source=json.loads(result.read_text(encoding='utf-8'))
    record={k:v for k,v in source.items() if k not in ('generated_texts','itls')}
    texts=source.get('generated_texts',[])
    record.update(schema_version=1, run_id=run_id,
        raw_result_sha256=sha(result.read_bytes()),
        config_sha256=sha(config.read_bytes()),
        generated_text_sha256=[sha(t.encode()) for t in texts],
        metric_notes={
            'output_throughput':'Total generated tokens / benchmark wall time; includes prefill and request overhead.',
            'mean_ttft_ms':'Client-observed mean time to first streamed token.',
            'mean_tpot_ms':'Mean per-request time per output token, excluding its first token.',
            'mean_itl_ms':'Stream event interval; MTP can emit multiple tokens per event.',
            'max_concurrent_requests':'CLI-derived estimate; use configured max_concurrency and server gauges for concurrency.'})
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result',type=Path)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.write_text(json.dumps(export(a.result,a.config,a.run_id),indent=2)+'\n',encoding='utf-8')
