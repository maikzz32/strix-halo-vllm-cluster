"""Bounded isolated M4 HC comparison; synthetic BF16 weights, no serving edits."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import statistics
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'


def pack(words):
    import numpy as np
    matrix=words.reshape(-1,256)
    exponent=(matrix>>7)&255
    base=exponent.min(axis=1)
    raw=exponent.max(axis=1)-base>15
    delta=exponent-base[:,None]
    codes=((matrix&127)|((matrix>>4)&2048)|(delta<<7)).astype(np.uint32)
    pairs=codes[:,::2]|(codes[:,1::2]<<12)
    packed=np.stack((pairs&255,(pairs>>8)&255,(pairs>>16)&255),axis=-1).astype(np.uint8).reshape(-1,384)
    lengths=np.where(raw,512,384).astype(np.uint32)
    offsets=np.concatenate((np.zeros(1,dtype=np.uint32),np.cumsum(lengths[:-1],dtype=np.uint32)))
    if int(offsets[-1])//16>=2**23:raise ValueError('packed tensor too large')
    desc=(base.astype(np.uint32)<<24)|(raw.astype(np.uint32)<<23)|(offsets//16)
    payload=b''.join(matrix[j].tobytes() if raw[j] else packed[j].tobytes() for j in range(len(matrix)))
    return desc,np.frombuffer(payload,dtype=np.uint8).copy()


def run(args):
    import numpy as np
    import torch
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(7211)
    lib=ctypes.CDLL(str(args.library))
    lib.hc_run.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    lib.hc_run.restype=ctypes.c_int
    lib.hc_unpack.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint,ctypes.c_void_p]
    lib.hc_unpack.restype=ctypes.c_int
    cu=torch.cuda.get_device_properties(0).multi_processor_count
    report={'status':'running','synthetic_weights':True,'pool_size':24,'cases':[],
            'device':torch.cuda.get_device_name(),'cu_count':cu,'torch':torch.__version__,
            'notes':['Same weights and arithmetic, only packed load sites differ.',
                     '24 distinct weights per shape exceed150MiB; cache misses not profiled.',
                     'Every packed matrix is unpacked on GPU and checked bit-exactly.',
                     'Both isolated GEMMs must equal installed vLLM outputs before timing.']}
    started=time.monotonic()
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    for label,n,k in [('merged_down',336,10240),('final_down',320,10240),('up',10240,320)]:
        pool=[]
        x=torch.randn((4,k),device='cuda',dtype=torch.bfloat16)
        for j in range(24):
            w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)*0.02
            if label=='merged_down':w[-12:]=0
            desc,data=pack(w.view(torch.int16).cpu().numpy().view(np.uint16))
            d=torch.from_numpy(desc.view(np.int32)).cuda();p=torch.from_numpy(data).cuda()
            decoded=torch.empty_like(w)
            rc=lib.hc_unpack(d.data_ptr(),p.data_ptr(),decoded.data_ptr(),w.numel(),torch.cuda.current_stream().cuda_stream)
            if rc:raise RuntimeError(('unpack launch',rc))
            torch.cuda.synchronize()
            if not torch.equal(w.view(torch.int16),decoded.view(torch.int16)):raise RuntimeError('GPU unpack mismatch')
            pool.append((w,d,p));del decoded
        report['cases'].append({'shape':[4,n,k],'label':label,'raw_bytes':sum(w.numel()*2 for w,d,p in pool),
                'packed_bytes':sum(d.numel()*4+p.numel() for w,d,p in pool),'gpu_unpack_exact':True,'methods':[]})
        case=report['cases'][-1];save()
        def call(method,item):
            w,d,p=item
            if method=='installed':return rocm_unquantized_gemm_impl(x,w)
            y=torch.empty((4,n),device='cuda',dtype=torch.bfloat16)
            rc=lib.hc_run({'reference':0,'packed':1,'reference_u1':3,'reference_u2':4}[method],w.data_ptr(),x.data_ptr(),y.data_ptr(),n,k,cu,torch.cuda.current_stream().cuda_stream,d.data_ptr(),p.data_ptr())
            if rc:raise RuntimeError(('GEMM launch',rc))
            return y
        for method in ['reference','packed','reference_u1','reference_u2']:
            mismatches=0
            for item in pool:
                expected=call('installed',item);actual=call(method,item)
                mismatches+=int((expected.view(torch.int16)!=actual.view(torch.int16)).sum())
            case[method+'_bit_mismatches']=mismatches;save()
            if mismatches:raise RuntimeError((label,method,'installed numerical mismatch',mismatches))
        stream=torch.cuda.Stream()
        for method in ['installed','reference','packed','reference_u1','reference_u2']:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for item in pool:call(method,item)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):outputs=[call(method,item) for item in pool]
                graph.replay();stream.synchronize()
                for item,y in zip(pool,outputs):
                    if not torch.equal(y,call('installed',item)):raise RuntimeError('captured output mismatch')
                original=x.clone();x.mul_(0.75)
                changed=[call('installed',item) for item in pool]
                graph.replay();stream.synchronize()
                if any(not torch.equal(y,e) for y,e in zip(outputs,changed)):raise RuntimeError('stale replay')
                x.copy_(original);graph.replay();stream.synchronize()
                samples=[]
                for sample in range(5):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for repeat in range(20):graph.replay()
                    end.record();end.synchronize();samples.append(begin.elapsed_time(end)*1000/(20*len(pool)))
                case['methods'].append({'method':method,'samples_us':samples,'median_us':statistics.median(samples),
                                        'changed_input_graph_exact':True});save()
                del graph,outputs,changed,original
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize();del pool,x,w,d,p,expected,actual,item;gc.collect();torch.cuda.empty_cache()
    report['status']='passed';report['elapsed_s']=time.monotonic()-started;save()
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
