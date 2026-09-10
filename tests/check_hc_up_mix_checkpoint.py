"""Isolated HC-up/gate fusion comparison, fixed original BF16 arithmetic target."""
import argparse,ctypes,hashlib,importlib,json,os,statistics
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
ap=argparse.ArgumentParser()
ap.add_argument('--seed',type=int,default=9567)
ap.add_argument('--library',required=True)
ap.add_argument('--sha256',required=True)
args=ap.parse_args()
import torch,triton,triton.language as tl
@triton.jit
def build_table(x,y,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 v=tl.load(x+i).to(tl.float32)
 tl.store(y+i,tl.sigmoid(v))
raw_bits=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
table=torch.empty(65536,device='cuda',dtype=torch.float32)
build_table[(256,)](raw_bits,table,256)
from vllm.model_executor.layers import utils
hc=importlib.import_module('vllm.models.qwen4_exp.amd.ops.hc')

torch.set_num_threads(1);torch.manual_seed(args.seed)
p=Path(args.library)
assert hashlib.sha256(p.read_bytes()).hexdigest()==args.sha256
lib=ctypes.CDLL(str(p));lib.hc_up_mix.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int,ctypes.c_void_p];lib.hc_up_mix.restype=ctypes.c_int
from safetensors import safe_open
root=Path('/home/cluster-user/qwen38_rest')
index_path=next(root.glob('*.index.json'))
index_bytes=index_path.read_bytes()
index=json.loads(index_bytes)['weight_map']
keys=sorted(k for k in index if k.startswith('model.language_model.') and k.endswith('input_mix_weight_up.weight'))
assert len(keys)==97,len(keys)
report=dict(status='running',seed=args.seed,binary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    index_sha256=hashlib.sha256(index_bytes).hexdigest(),cases=[],failures=[])
x=torch.empty(4,320,device='cuda',dtype=torch.bfloat16)
xn=torch.empty(4,10240,device='cuda',dtype=torch.bfloat16)
y=torch.empty(4,2560,device='cuda',dtype=torch.bfloat16)
gates=torch.empty(4,10240,device='cuda',dtype=torch.bfloat16)
for key in keys:
    with safe_open(str(root/index[key]),framework='pt',device='cpu') as f:
        cpu=f.get_tensor(key)
    assert tuple(cpu.shape)==(10240,320) and cpu.dtype==torch.bfloat16
    weight_sha=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    w=cpu.to('cuda')
    for pattern in ('independent','shared_rows','positive'):
        z=torch.randn(4,320,device='cuda',dtype=torch.bfloat16)
        if pattern=='shared_rows':z.copy_(z[:1].clone().expand_as(z))
        if pattern=='positive':z.abs_()
        xn.normal_()
        for scale in (.02,.2,1.,8.):
            x.copy_(z*scale)
            expected_gates=utils.rocm_unquantized_gemm_impl(x,w)
            expected=hc.hc_gate_mix(xn,expected_gates,4)
            rc=lib.hc_up_mix(w.data_ptr(),x.data_ptr(),xn.data_ptr(),y.data_ptr(),gates.data_ptr(),table.data_ptr(),60,torch.cuda.current_stream().cuda_stream)
            assert rc==0,rc
            ng=int((gates!=expected_gates).sum());ny=int((y!=expected).sum())
            case=dict(key=key,weight_sha256=weight_sha,pattern=pattern,scale=scale,gate_unequal=ng,output_unequal=ny)
            report['cases'].append(case)
            if ng or ny:
                report['failures'].append(case)
                # Report counts only; no checkpoint tensor content is exported.
                print('FAILURE='+json.dumps(case),flush=True)
    del w,cpu
report['status']='parity_failed' if report['failures'] else 'passed'
print('RESULT='+json.dumps(report),flush=True)
