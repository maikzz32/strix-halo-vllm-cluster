"""Opt-in output alias for the pinned TP4 WNA16 path; no arithmetic changes."""
import argparse
import hashlib
from pathlib import Path

BASE_SHA = '1e60aca6ed0dd4fcb46d577897ff1651f27a6130b3449d22265c0c791beec5d5'
OLD = '''            if use_output_alias and rocm_aiter_ops.is_fused_moe_enabled():
                fused_out = output_alias'''
NEW = '''            # STRIX_MOE_OUTPUT_ALIAS: caller allocates a fresh output in apply().
            # Restrict to the audited small-batch TP4 path, without enabling AITER.
            import os
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
            from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
            from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
            strix_alias = (
                os.environ.get("STRIX_MOE_OUTPUT_ALIAS", "0") == "1"
                and type(self.fused_experts) is TritonWNA16Experts
                and type(self.prepare_finalize) is MoEPrepareAndFinalizeNoDPEPModular
                and type(self.fused_experts.finalize_weight_and_reduce_impl()) is TopKWeightAndReduceNoOP
                and in_dtype == torch.bfloat16
                and 0 < M_full <= 8
                and tuple(w1.shape) == (512, 320, 1280)
                and tuple(w2.shape) == (512, 2560, 80)
                and expert_map is None
                and not self.fused_experts.moe_config.is_lora_enabled
                and self.fused_experts.quant_config.use_int4_w4a16
            )
            if use_output_alias and (rocm_aiter_ops.is_fused_moe_enabled() or strix_alias):
                fused_out = output_alias
                if strix_alias:
                    logger.info_once("STRIX_MOE_OUTPUT_ALIAS active: WNA16 TP4 small batch")'''

def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError('Unexpected modular kernel source SHA')
    if source.count(OLD) != 1:
        raise ValueError('Expected exactly one alias anchor')
    result = source.replace(OLD, NEW)
    compile(result, '<modular-kernel>', 'exec')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    before = a.source.read_text(encoding='utf-8')
    after = transform(before)
    print('before', BASE_SHA, 'after', hashlib.sha256(after.encode()).hexdigest())
    if a.apply:
        backup = a.source.with_name(a.source.name + '.pre-strix-alias-' + BASE_SHA)
        if backup.exists() and backup.read_text(encoding='utf-8') != before:
            raise ValueError('Backup mismatch')
        backup.write_text(before, encoding='utf-8')
        temporary = a.source.with_name(a.source.name + '.strix-alias-tmp')
        temporary.write_text(after, encoding='utf-8')
        temporary.replace(a.source)
