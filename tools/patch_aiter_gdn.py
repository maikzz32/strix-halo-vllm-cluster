"""Build an opt-in packed GDN speculative-path patch against a pinned runtime."""
import hashlib

BASE_SHA = 'cb819264c1e791177d866f484ca0579a7d57141926705a1f75dd4a79de4d0125'


def patched(data):
    assert hashlib.sha256(data).hexdigest() == BASE_SHA, 'Unexpected GDN source'
    source = data.decode()
    anchor = 'logger = init_logger(__name__)'
    imports = '''STRIX_PACKED_GDN = os.environ.get("STRIX_PACKED_GDN", "0") == "1"
if STRIX_PACKED_GDN:
    from vllm.model_executor.layers.mamba.gdn._strix_gdn import (
        fused_rearrange_sigmoid_gated_delta_rule as strix_packed_gdn,
    )

'''
    assert source.count(anchor) == 1
    source = source.replace(anchor, imports + anchor)
    anchor = '        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)'
    replacement = '''        use_strix_packed_gdn = (
            STRIX_PACKED_GDN
            and spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
            and 0 < attn_metadata.num_spec_decodes <= 2
            and self.head_k_dim == 128
            and self.head_v_dim == 128
            and self.key_dim // self.tp_size == 512
            and self.value_dim // self.tp_size == 1536
            and ssm_state.dtype in (torch.float32, torch.bfloat16)
            and mixed_qkv_spec.dtype == torch.bfloat16
            and mixed_qkv_spec.shape[0] == 4 * attn_metadata.num_spec_decodes
            and ssm_state.shape[1:] == (12, 128, 128)
            and ssm_state.stride()[1:] == (16384, 128, 1)
            and spec_state_indices_tensor.stride(-1) == 1
        )
        if not use_strix_packed_gdn:
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)'''
    assert source.count(anchor) == 1
    source = source.replace(anchor, replacement)
    anchor = '''        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:'''
    replacement = '''        # 2.1: Process the multi-query part
        if use_strix_packed_gdn:
            core_attn_out_spec, last_recurrent_state = strix_packed_gdn(
                A_log=self.A_log,
                a=a_spec,
                b=b_spec,
                dt_bias=self.dt_bias,
                qkv=mixed_qkv_spec,
                key_dim=self.key_dim // self.tp_size,
                value_dim=self.value_dim // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[:attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        elif spec_sequence_masks is not None:'''
    assert source.count(anchor) == 1
    source = source.replace(anchor, replacement)
    compile(source, 'qwen_gdn_linear_attn.py', 'exec')
    return source.encode()


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = args.base_source.read_bytes()
    files = {
        'original.py': original,
        'candidate.py': patched(original),
        '_strix_gdn.py': (root/'tests/aiter_gdn_wrapper.py').read_text().encode(),
        '_strix_gdn_kernel.py': (root/'tests/aiter_gdn_null_guard.py').read_text().encode(),
    }
    for name, data in files.items():
        compile(data, name, 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (args.output/name).write_bytes(data)
    manifest = {'base_sha': BASE_SHA,
        'files': {n: hashlib.sha256(b).hexdigest() for n,b in files.items()},
        'flag': 'STRIX_PACKED_GDN=1', 'activated': False}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
