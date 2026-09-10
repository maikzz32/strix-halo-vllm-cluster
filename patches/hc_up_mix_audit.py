"""Diagnostic HC pair: return baseline output, retain the first M4 mismatch.

Requires explicit deployment. No background thread or remote control endpoint.
CPU collection must occur only after the caller makes the model idle.
"""
import torch
import hc_up_mix_dynamic as native
from hc_first_mismatch import inspect, SNAPSHOT_ELEMENTS
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.utils.torch_utils import direct_register_custom_op
_REGISTERED=False
LAYERS={}

def operation(x:torch.Tensor,w:torch.Tensor,xn:torch.Tensor,table:torch.Tensor,
              state:torch.Tensor,snapshot:torch.Tensor)->torch.Tensor:
    gates=rocm_unquantized_gemm_impl(x,w)
    output=torch.ops.vllm.qwen4_exp_hc_gate_mix(xn,gates,4)
    eligible=(tuple(x.shape)==(4,320) and tuple(w.shape)==(10240,320)
              and tuple(xn.shape)==(4,10240) and table.shape==(65536,)
              and x.dtype==w.dtype==xn.dtype==torch.bfloat16
              and table.dtype==torch.float32 and x.is_cuda
              and x.device==w.device==xn.device==table.device
              and all(t.is_contiguous() for t in (x,w,xn,table)))
    if eligible:
        proposed=torch.empty_like(output);pg=torch.empty_like(gates)
        with torch.cuda.device(x.device):
            rc=native._LIBRARY.hc_up_mix(w.data_ptr(),x.data_ptr(),xn.data_ptr(),
                proposed.data_ptr(),pg.data_ptr(),table.data_ptr(),60,
                torch.cuda.current_stream(x.device).cuda_stream)
        if rc:raise RuntimeError(('HC audit launch',rc))
        inspect(x,xn,gates,pg,output,proposed,state,snapshot)
    return output

def fake(x:torch.Tensor,w:torch.Tensor,xn:torch.Tensor,table:torch.Tensor,
         state:torch.Tensor,snapshot:torch.Tensor)->torch.Tensor:
    return x.new_empty((*x.shape[:-1],2560))

def prepare(module):
    global _REGISTERED
    native.prepare(module)
    if not module._strix_up_mix_selected:return
    if not _REGISTERED:
        direct_register_custom_op(op_name='strix_hc_up_mix_audit',op_func=operation,
            fake_impl=fake,mutates_args=['state','snapshot'])
        _REGISTERED=True
    device=module.input_mix_weight_up.weight.device
    module.register_buffer('_strix_audit_state',torch.zeros(4,device=device,dtype=torch.int32),persistent=False)
    module.register_buffer('_strix_audit_snapshot',torch.empty(SNAPSHOT_ELEMENTS,device=device,dtype=torch.bfloat16),persistent=False)
    prefix=module.input_mix_weight_up.prefix
    if prefix in LAYERS:raise RuntimeError('Duplicate audited HC prefix')
    LAYERS[prefix]=module

def mix(module,x,xn,original_gate_mix):
    if module._strix_up_mix_selected:
        return torch.ops.vllm.strix_hc_up_mix_audit(x,module.input_mix_weight_up.weight,
            xn,module._strix_up_mix_table,module._strix_audit_state,module._strix_audit_snapshot)
    return original_gate_mix(xn,module.input_mix_weight_up(x),module.hc_count)
