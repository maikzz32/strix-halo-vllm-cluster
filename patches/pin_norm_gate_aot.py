"""Apply the norm geometry pin when AOT supplies bundled Triton kernels."""
from pin_norm_gate_geometry import BODY_SHA256, body_hash
from pin_norm_gate_geometry import install as install_geometry

_INSTALLED = False
_SOURCE_TAG = '\n# strix_norm_geometry_xblock4_aot_v2\n'


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    install_geometry()
    from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels
    from torch._inductor import config
    original = AsyncCompile.triton

    def compile_kernel(self, kernel_name, source_code, device_str='cuda'):
        if body_hash(source_code) != BODY_SHA256 or not (
            "'gfx1151'" in source_code or '"gfx1151"' in source_code
        ):
            return original(self, kernel_name, source_code, device_str)
        # A distinct source cache key excludes the old bundled launcher.
        # Compile this kernel in the process containing the geometry hook.
        tagged_source = source_code + _SOURCE_TAG
        CompiledTritonKernels.remove_future(tagged_source)
        with config.patch(compile_threads=1):
            return original(self, kernel_name, tagged_source, device_str)

    AsyncCompile.triton = compile_kernel
    _INSTALLED = True
