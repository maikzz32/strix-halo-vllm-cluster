"""Diagnostic custom-op boundary around the unchanged HC pair.

No native fusion, lookup table, extra tensor buffers, or arithmetic changes.
This is an isolation control, not a performance candidate.
"""
import re
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

_REGISTERED = False
_PREFIX = re.compile(r"(?:^|\.)(?:attn_hyper_connection|mlp_hyper_connection|hyper_connection_mixer)\.input_mix_weight_up$")
_EXCLUDE = re.compile(r"(?:^|\.)(?:mtp|visual|vision_model)(?:\.|$)")


def _operation(x: torch.Tensor, w: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
    gates = torch.ops.vllm.rocm_unquantized_gemm(x, w, None)
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(xn, gates, 4)


def _fake(x: torch.Tensor, w: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 2560))


def prepare(module):
    global _REGISTERED
    layer = module.input_mix_weight_up
    prefix = getattr(layer, "prefix", "")
    module._strix_up_mix_selected = bool(
        _PREFIX.search(prefix) and not _EXCLUDE.search(prefix)
        and isinstance(layer.quant_method, UnquantizedLinearMethod)
        and module.hc_count == 4 and module.hidden_size == 2560
        and module.lora_rank == 320 and layer.bias is None
    )
    if module._strix_up_mix_selected and not _REGISTERED:
        direct_register_custom_op(op_name="strix_hc_up_mix_reference",
                                  op_func=_operation, fake_impl=_fake)
        _REGISTERED = True


def mix(module, x, xn, original_gate_mix):
    if module._strix_up_mix_selected:
        return torch.ops.vllm.strix_hc_up_mix_reference(
            x, module.input_mix_weight_up.weight, xn)
    return original_gate_mix(xn, module.input_mix_weight_up(x), module.hc_count)
