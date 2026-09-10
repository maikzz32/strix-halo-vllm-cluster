"""HC fusion with shape dispatch inside the custom operation for dynamic graphs."""
import ctypes
import hashlib
import os
import re
from pathlib import Path
import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

BINARY_SHA256 = "a3b1e46f38a01da4a4ad2fe3647573ca9fc4f0042fd908193c2a6f79b93f5e30"
_TABLES = {}
_LIBRARY = None
_REGISTERED = False
_PREFIX = re.compile(r"(?:^|\.)(?:attn_hyper_connection|mlp_hyper_connection|hyper_connection_mixer)\.input_mix_weight_up$")
_EXCLUDE = re.compile(r"(?:^|\.)(?:mtp|visual|vision_model)(?:\.|$)")

@triton.jit
def _build_table(bits, table, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(table + i, tl.sigmoid(tl.load(bits + i).to(tl.float32)))


def _operation(x: torch.Tensor, w: torch.Tensor, xn: torch.Tensor,
               table: torch.Tensor) -> torch.Tensor:
    if not (tuple(x.shape) == (4, 320) and tuple(w.shape) == (10240, 320)
            and tuple(xn.shape) == (4, 10240) and tuple(table.shape) == (65536,)
            and x.dtype == w.dtype == xn.dtype == torch.bfloat16
            and table.dtype == torch.float32 and x.is_cuda
            and x.device == w.device == xn.device == table.device
            and all(t.is_contiguous() for t in (x, w, xn, table))):
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
        return torch.ops.vllm.qwen4_exp_hc_gate_mix(
            xn, rocm_unquantized_gemm_impl(x, w), 4)
    y = x.new_empty((4, 2560))
    with torch.cuda.device(x.device):
        rc = _LIBRARY.hc_up_mix(w.data_ptr(), x.data_ptr(), xn.data_ptr(),
            y.data_ptr(), None, table.data_ptr(), 60,
            torch.cuda.current_stream(x.device).cuda_stream)
    if rc:
        raise RuntimeError(("HC fusion launch failed", rc))
    return y


def _fake(x: torch.Tensor, w: torch.Tensor, xn: torch.Tensor,
          table: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 2560))


def prepare(module):
    """Called during HC construction, before compile/capture or model execution."""
    global _LIBRARY, _REGISTERED
    layer = module.input_mix_weight_up
    prefix = getattr(layer, "prefix", "")
    selected = bool(_PREFIX.search(prefix)) and not _EXCLUDE.search(prefix)
    selected = (selected and isinstance(layer.quant_method, UnquantizedLinearMethod)
                and module.hc_count == 4 and module.hidden_size == 2560
                and module.lora_rank == 320 and layer.bias is None)
    module._strix_up_mix_selected = bool(selected)
    if not selected:
        return
    w = layer.weight
    if not w.is_cuda or w.dtype != torch.bfloat16 or tuple(w.shape) != (10240, 320):
        raise RuntimeError("HC fusion requires initialized CUDA BF16 parameter storage")
    if _LIBRARY is None:
        p = Path(os.environ["STRIX_HC_UP_MIX_LIBRARY"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != BINARY_SHA256:
            raise ValueError("HC fusion library SHA256 mismatch")
        _LIBRARY = ctypes.CDLL(str(p))
        _LIBRARY.hc_up_mix.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int, ctypes.c_void_p]
        _LIBRARY.hc_up_mix.restype = ctypes.c_int
    if not _REGISTERED:
        direct_register_custom_op(op_name="strix_hc_up_mix", op_func=_operation,
                                  fake_impl=_fake)
        _REGISTERED = True
    key = w.device
    if key not in _TABLES:
        with torch.cuda.device(key):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HC table initialization during capture")
            bits = torch.arange(65536, device=key, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            table = torch.empty(65536, device=key, dtype=torch.float32)
            _build_table[(256,)](bits, table, 256)
            torch.cuda.current_stream(key).synchronize()
            _TABLES[key] = table
    module.register_buffer("_strix_up_mix_table", _TABLES[key], persistent=False)


def mix(module, x, xn, original_gate_mix):
    w = module.input_mix_weight_up.weight
    if module._strix_up_mix_selected:
        return torch.ops.vllm.strix_hc_up_mix(x, w, xn, module._strix_up_mix_table)
    return original_gate_mix(xn, module.input_mix_weight_up(x), module.hc_count)
