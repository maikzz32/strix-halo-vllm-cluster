"""Default-off native HC geometry experiment; original BF16 weights retained."""
import ctypes
import hashlib
import os
from pathlib import Path
import re

LIB_SHA='8ab4de1bd05abaf4f0188a1e82aa6a1f69e53acb8ed2cc63444e5f06b6b854bc'
_PREFIX=re.compile(r'(?:^|\.)(?:attn_hyper_connection|mlp_hyper_connection|hyper_connection_mixer)\.(input_mix_weight_down_block_inject|input_mix_weight_down|input_mix_weight_up)$')
_INSTALLED=False

def install():
    global _INSTALLED
    if os.environ.get('STRIX_HC_NATIVE_GRID','0')!='1':return False
    if _INSTALLED:return True
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op
    if not torch.version.hip:raise RuntimeError('ROCm required')
    path=Path(os.environ['STRIX_HC_NATIVE_LIBRARY'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=LIB_SHA:raise ValueError('Unexpected native library SHA')
    lib=ctypes.CDLL(str(path))
    lib.hc_run.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    lib.hc_run.restype=ctypes.c_int
    choices={(336,10240):(0,40),(320,10240):(0,30),(10240,320):(4,20)}
    def operation(x:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
        n,k=weight.shape
        mode,grid=choices[(n,k)]
        y=x.new_empty((*x.shape[:-1],n))
        # The pinned library's explicit mode 4 uses the original BF16 path.
        # Descriptor arguments are unused there, but its common ABI rejects nulls.
        with torch.cuda.device(x.device):
            rc=lib.hc_run(mode,weight.data_ptr(),x.data_ptr(),y.data_ptr(),n,k,grid,
                torch.cuda.current_stream(x.device).cuda_stream,weight.data_ptr(),weight.data_ptr())
        if rc:raise RuntimeError(('HC native launch',mode,grid,rc))
        return y
    def fake(x:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
        return x.new_empty((*x.shape[:-1],weight.shape[0]))
    direct_register_custom_op(op_name='strix_hc_native_grid',op_func=operation,fake_impl=fake)
    original=UnquantizedLinearMethod.apply
    def apply(self,layer,x,bias=None):
        prefix=getattr(layer,'prefix','');w=layer.weight
        selected=(_PREFIX.search(prefix) is not None and
                  not re.search(r'(?:^|\.)(?:mtp|visual|vision_model)(?:\.|$)',prefix))
        if (selected and bias is None and x.ndim>=2 and w.ndim==2
            and x.numel()//x.shape[-1]==4 and tuple(w.shape) in choices
            and x.shape[-1]==w.shape[1] and x.dtype==w.dtype==torch.bfloat16
            and x.is_cuda and x.device==w.device and x.is_contiguous() and w.is_contiguous()):
            return torch.ops.vllm.strix_hc_native_grid(x,w)
        return original(self,layer,x,bias)
    UnquantizedLinearMethod.apply=apply
    _INSTALLED=True
    print('STRIX_HC_NATIVE_GRID installed: target HC M4 BF16, pinned native library',flush=True)
    return True
