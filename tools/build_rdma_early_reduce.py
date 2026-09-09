"""Overlap the unchanged host reduction with outstanding local send completions."""


def generate(source):
    start = source.index('static int collective(')
    end = source.index('\nstatic void copy_error(', start)
    body = source[start:end]
    reduction = '''    const uint16_t *a[NR];
    for (int r = 0; r < NR; ++r) a[r] = r == s->rank ? input : s->memory + (slot * NR + r) * N;
    if (cpu_rdma_reduce_host(a, output, N, (cpu_rdma_mode)mode) != CPU_RDMA_OK) return -1;'''
    assert body.count(reduction) == 1
    body = body.replace(reduction, '    if (!reduced) return -1;')
    marker = '    unsigned polls = 0;'
    assert body.count(marker) == 1
    body = body.replace(marker, '    int reduced = 0;\n' + marker)
    marker = '    }\n    /* Two slots suffice:'
    assert body.count(marker) == 1
    # All current receives have completed; later receives target the other slot.
    # The original loop still waits for every local send before returning.
    block = '''        if (!reduced && s->recv_mask[slot] == full) {
            atomic_thread_fence(memory_order_acquire);
REDUCTION
            reduced = 1;
        }
'''.replace('REDUCTION', '\n'.join('        ' + line for line in reduction.splitlines()))
    body = body.replace(marker, block + marker)
    return source[:start] + body + source[end:]
