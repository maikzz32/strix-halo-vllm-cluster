"""Isolated sparse GQA empty-tile screening; never edits serving modules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile

import torch
from vllm.models.qwen4_exp.amd.ops import qsa as original


def main():
    torch.set_num_threads(1)
    assert 'gfx1151' in torch.cuda.get_device_properties(0).gcnArchName
    source = Path(original.__file__).read_text()
    start = source.index('        # physical_page * block stride can overflow int32')
    end = source.index('\n    has_values = normalizer > 0', start)
    body = source[start:end]
    candidate_source = source[:start] + '        if tl.sum(valid.to(tl.int32), axis=0) > 0:\n' + ''.join('    ' + line + '\n' for line in body.splitlines()) + source[end:]
    with tempfile.TemporaryDirectory(prefix='qsa-sparse-empty-') as folder:
        path = Path(folder) / 'candidate.py'
        path.write_text(candidate_source)
        spec = importlib.util.spec_from_file_location('qsa_sparse_empty_candidate', path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        records = []
        torch.manual_seed(9341)
        # TP4: 24 query heads / 4, replicated KV head, head dimension 256.
        cache_k = torch.randn((64, 64, 1, 256), device='cuda', dtype=torch.bfloat16)
        cache_v = torch.randn_like(cache_k)
        table = torch.arange(64, device='cuda', dtype=torch.int32)[None, :]
        for rows in (4, 8, 504, 1024):
            q = torch.randn((rows, 6, 256), device='cuda', dtype=torch.bfloat16)
            req = torch.zeros(rows, device='cuda', dtype=torch.int32)
            indices = torch.empty((rows, 2051), device='cuda', dtype=torch.int32)
            outputs = [torch.empty_like(q), torch.empty_like(q)]
            calls = [lambda m=m, out=out: m.qsa_sparse_paged_attention(q, cache_k, cache_v, indices, table, req, out) for m, out in zip((original, candidate), outputs)]
            graphs = []
            indices.fill_(-1)
            for call in calls:
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                graphs.append(graph)
            for pattern in ('empty', 'causal', 'holes', 'full', 'invalid_page'):
                columns = torch.arange(2051, device='cuda', dtype=torch.int32)[None, :].expand(rows, -1)
                indices.copy_(columns)
                req.zero_()
                if pattern == 'empty':
                    indices.fill_(-1)
                elif pattern == 'causal':
                    indices.masked_fill_(columns >= torch.arange(1, rows + 1, device='cuda')[:, None], -1)
                elif pattern == 'holes':
                    indices.masked_fill_((columns // 64) % 2 == 0, -1)
                elif pattern == 'invalid_page':
                    indices[:, 64:128] = 100000
                    req[0] = -1
                # Input values change after capture, including masks and request IDs.
                q.normal_()
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                differences = int((outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).sum().item())
                assert differences == 0, (rows, pattern, differences)
                eager = calls[1]().clone()
                graphs[1].replay()
                torch.cuda.synchronize()
                assert torch.equal(eager.view(torch.int16), outputs[1].view(torch.int16))
                samples = [[], []]
                for repeat in range(6):
                    for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(8):
                            graphs[i].replay()
                        finish.record()
                        finish.synchronize()
                        samples[i].append(begin.elapsed_time(finish) * 1000 / 8)
                records.append(dict(rows=rows, pattern=pattern, differences=differences, changed_graph_exact=True, samples_us=samples, medians_us=[statistics.median(x) for x in samples]))
        print(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(), candidate_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(), query_heads=6, head_dim=256, selection_width=2051, records=records)), flush=True)


if __name__ == '__main__':
    main()
