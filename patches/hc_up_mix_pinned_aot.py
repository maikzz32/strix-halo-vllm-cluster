"""HC fusion with the original norm/gate launch geometry retained."""
from pin_norm_gate_aot import install

install()

from hc_up_mix_dynamic import prepare, mix  # noqa: E402,F401
