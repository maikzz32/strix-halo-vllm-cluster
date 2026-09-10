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
x=torch.randn(4,320,device='cuda',dtype=torch.bfloat16)*.2
xn=torch.randn(4,10240,device='cuda',dtype=torch.bfloat16)
weights=[torch.randn(10240,320,device='cuda',dtype=torch.bfloat16)*.02 for _ in range(12)]
gates=torch.empty(4,10240,device='cuda',dtype=torch.bfloat16)
def call(method,w):
 if method=='installed':return hc.hc_gate_mix(xn,utils.rocm_unquantized_gemm_impl(x,w),4)
 y=torch.empty(4,2560,device='cuda',dtype=torch.bfloat16)
 assert lib.hc_up_mix(w.data_ptr(),x.data_ptr(),xn.data_ptr(),y.data_ptr(),gates.data_ptr(),table.data_ptr(),int(method),torch.cuda.current_stream().cuda_stream)==0
 return y
methods=['installed','20','30','40','60'];report=dict(status='running',seed=args.seed,checks=[],binary_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for method in methods[1:]:
 differences=[];gate_differences=[]
 for w in weights:
  a=call('installed',w);b=call(method,w);differences.append(int((a!=b).sum()));gate_differences.append(int((gates!=utils.rocm_unquantized_gemm_impl(x,w)).sum()))
 report['checks'].append(dict(method=method,unequal_elements=differences,gate_unequal_elements=gate_differences))
if any(any(r['unequal_elements']) or any(r['gate_unequal_elements']) for r in report['checks']):
 report['status']='parity_failed';print('RESULT='+json.dumps(report),flush=True);raise SystemExit(1)
# Exclude the diagnostic intermediate stores from timing.
gates_pointer=None
original_call=call
def call(method,w):
 if method=='installed':return original_call(method,w)
 y=torch.empty(4,2560,device='cuda',dtype=torch.bfloat16)
 assert lib.hc_up_mix(w.data_ptr(),x.data_ptr(),xn.data_ptr(),y.data_ptr(),None,table.data_ptr(),int(method),torch.cuda.current_stream().cuda_stream)==0
 return y
# Cover a wider gate range and zero activations before timing.
saved_x=x.clone();saved_xn=xn.clone()
for factor in (0.0,4.0,32.0):
 x.copy_(saved_x*factor)
 for w in weights[:4]:
  expected=call('installed',w)
  for method in methods[1:]:assert torch.equal(expected,call(method,w)),('range',factor,method)
x.copy_(saved_x);xn.copy_(saved_xn)
report['range_factors_exact']=[0.0,4.0,32.0]
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
 graphs={};outputs={}
 for method in methods:
  for w in weights:call(method,w)
  stream.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):out=[call(method,w) for w in weights]
  graphs[method]=g;outputs[method]=out
 x.mul_(.5);xn.add_(.0625);expected=[call('installed',w) for w in weights]
 for g in graphs.values():g.replay()
 stream.synchronize();assert all(torch.equal(a,b) for out in outputs.values() for a,b in zip(expected,out))
 samples={m:[] for m in methods}
 for r in range(6):
  for m in (methods if r%2==0 else methods[::-1]):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(20):graphs[m].replay()
   b.record();b.synchronize();samples[m].append(a.elapsed_time(b)*1000/240)
report.update(status='passed',changed_graph_exact=True,samples_us=samples,median_us={m:statistics.median(v) for m,v in samples.items()})
print('RESULT='+json.dumps(report),flush=True)
