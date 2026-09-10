"""Whole-head recurrence feasibility before output-norm fusion; no model patch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import torch
import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as stock

torch.set_num_threads(1)
source = Path(stock.__file__).read_text()
sha = hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest()
assert sha == '000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f'
old = 'min(triton.next_power_of_2(V), 32)'
assert source.count(old) == 1
path = Path(tempfile.mkdtemp(prefix='gdn-whole-head-')) / 'candidate.py'
path.write_text(source.replace(old, 'min(triton.next_power_of_2(V), 128)'))
spec = importlib.util.spec_from_file_location('gdn_whole_head_candidate', path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
rows = []
for seed in (9611, 9612, 9613, 9614):
    torch.manual_seed(seed)
    q = torch.randn(1, 4, 4, 128, device='cuda', dtype=torch.bfloat16) * .25
    k = torch.randn_like(q)
    v = torch.randn(1, 4, 12, 128, device='cuda', dtype=torch.bfloat16)
    a = torch.randn(4, 12, device='cuda', dtype=torch.bfloat16)
    b = torch.randn_like(a)
    alog = torch.randn(12, device='cuda')
    bias = torch.randn_like(alog)
    initial = torch.randn(8, 12, 128, 128, device='cuda') * .02
    states = [initial.clone(), initial.clone()]
    indices = torch.tensor([[1, 3, 2, 4]], device='cuda', dtype=torch.int32)
    cu = torch.tensor([0, 4], device='cuda', dtype=torch.int32)
    accepted = torch.ones(1, device='cuda', dtype=torch.int32)
    def call(i):
        module = (stock, candidate)[i]
        return module.fused_sigmoid_gating_delta_rule_update(
            A_log=alog, a=a, b=b, dt_bias=bias, q=q, k=k, v=v,
            initial_state=states[i], inplace_final_state=True, cu_seqlens=cu,
            ssm_state_indices=indices, num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True)[0]
    for count in (1, 2, 3, 4):
        accepted.fill_(count)
        for state in states:
            state.copy_(initial)
        output = [call(i) for i in (0, 1)]
        torch.cuda.synchronize()
        rows.append(dict(seed=seed, accepted=count,
                         output_differences=int(torch.count_nonzero(output[0] != output[1])),
                         state_differences=int(torch.count_nonzero(states[0] != states[1]))))
    if seed == 9611:
        graphs, output = [], []
        for i in (0, 1):
            states[i].copy_(initial)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = call(i)
            stream.synchronize()
            graphs.append(graph)
            output.append(result)
        q.mul_(.5)
        accepted.fill_(2)
        for state in states:
            state.copy_(initial)
        expected = [call(i).clone() for i in (0, 1)]
        expected_states = [state.clone() for state in states]
        for state in states:
            state.copy_(initial)
        torch.cuda.synchronize()
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(output[i], expected[i]) and
                   torch.equal(states[i], expected_states[i]) for i in (0, 1))
        times = [[], []]
        # Restore recurrent state outside every timed replay.
        accepted.fill_(1)
        for state in states:
            state.copy_(initial)
        torch.cuda.synchronize()
        for round_id in range(8):
            for i in ((0, 1) if round_id % 2 == 0 else (1, 0)):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                samples = []
                for _ in range(32):
                    states[i].copy_(initial)
                    torch.cuda.synchronize()
                    start.record()
                    graphs[i].replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                times[i].append(statistics.median(samples))
        for graph in graphs:
            graph.reset()
print('RESULT=' + json.dumps(dict(status='passed', cases=rows,
    source_sha256=sha, candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    changed_input_graph_exact=True, samples_us=times,
    medians_us=[statistics.median(x) for x in times],
    note='BV32 versus BV128, same four warps. No output normalization fused.')))
