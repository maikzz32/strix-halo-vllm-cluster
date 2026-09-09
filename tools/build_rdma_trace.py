"""Generate opt-in bounded CPU collective tracing without changing arithmetic."""
from pathlib import Path
import argparse

DECL = r'''
#include <sys/mman.h>
enum { TRACE_CAPACITY = 2048 };
struct rdma_trace_record {
    uint64_t sequence, start_ns, posted_ns, ready_ns, reduced_ns, entry_recv_mask;
    uint64_t recv_ns[4], send_ns[4], polls, reserved;
};
struct rdma_trace_map {
    uint64_t magic, version, rank, capacity, requested, published, first_sequence, reserved;
    struct rdma_trace_record records[TRACE_CAPACITY];
};
_Static_assert(sizeof(struct rdma_trace_record) == 128, "trace record ABI");
_Static_assert(offsetof(struct rdma_trace_map, records) == 64, "trace header ABI");
static uint64_t trace_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
'''

HELPERS = r'''
static int trace_initialize(struct cpu_rdma_ctx *s) {
    const char *directory = getenv("STRIX_RDMA_TRACE_DIR");
    if (!directory || !*directory) return 0;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/rdma-r%d-p%ld.bin", directory, s->rank, (long)getpid());
    if (n < 0 || (size_t)n >= sizeof(path)) return -1;
    int fd = open(path, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    if (ftruncate(fd, sizeof(struct rdma_trace_map))) { close(fd); return -1; }
    void *mapping = mmap(NULL, sizeof(struct rdma_trace_map), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return -1;
    s->trace = mapping;
    memset(s->trace, 0, sizeof(*s->trace));
    s->trace->magic = 0x31524354414d4452ull;
    s->trace->version = 1; s->trace->rank = (uint64_t)s->rank;
    s->trace->capacity = TRACE_CAPACITY;
    const char *initial = getenv("STRIX_RDMA_TRACE_AUTOSTART");
    if (initial && *initial) {
        char *end = NULL; unsigned long amount = strtoul(initial, &end, 10);
        if (*end || amount > TRACE_CAPACITY) return -1;
        __atomic_store_n(&s->trace->requested, amount, __ATOMIC_RELEASE);
    }
    return 0;
}
static struct rdma_trace_record *trace_begin(struct cpu_rdma_ctx *s, uint32_t seq) {
    if (!s->trace) return NULL;
    uint64_t limit = __atomic_load_n(&s->trace->requested, __ATOMIC_ACQUIRE);
    uint64_t count = __atomic_load_n(&s->trace->published, __ATOMIC_RELAXED);
    if (!limit || limit > TRACE_CAPACITY || count >= limit) return NULL;
    struct rdma_trace_record *row = &s->trace->records[count];
    row->sequence = seq; row->start_ns = trace_now_ns();
    row->entry_recv_mask = s->recv_mask[seq & 1u];
    if (!count) s->trace->first_sequence = seq;
    return row;
}
static void trace_publish(struct cpu_rdma_ctx *s, struct rdma_trace_record *row) {
    if (!row) return;
    uint64_t count = (uint64_t)(row - s->trace->records) + 1;
    __atomic_store_n(&s->trace->published, count, __ATOMIC_RELEASE);
}
'''


def generate(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Unexpected transport source at ' + old[:70])
        source = source.replace(old, new)
    replace('struct cpu_rdma_ctx {', DECL + '\nstruct cpu_rdma_ctx {\n    struct rdma_trace_map *trace;\n    uint64_t trace_recv_ns[2][4];')
    replace('static int expired(', HELPERS + '\nstatic int expired(')
    replace('    atomic_thread_fence(memory_order_release);\n    for (int r = 0; r < NR;',
            '    struct rdma_trace_record *tr = trace_begin(s, seq);\n    atomic_thread_fence(memory_order_release);\n    for (int r = 0; r < NR;')
    replace('    unsigned polls = 0;', '    if (tr) tr->posted_ns = trace_now_ns();\n    unsigned polls = 0;')
    replace('        for (int j = 0; j < count; ++j) {',
            '        uint64_t batch_ns = tr && count > 0 ? trace_now_ns() : 0;\n        for (int j = 0; j < count; ++j) {')
    replace('                sent |= 1u << r;', '                sent |= 1u << r;\n                if (tr) tr->send_ns[r] = batch_ns;')
    replace('                s->recv_next[r]++;', '                if (tr) s->trace_recv_ns[got & 1u][r] = batch_ns;\n                s->recv_next[r]++;')
    replace('    const uint16_t *a[NR];',
            '    if (tr) { tr->ready_ns = trace_now_ns(); tr->polls = polls;\n        memcpy(tr->recv_ns, s->trace_recv_ns[slot], sizeof(tr->recv_ns)); }\n    const uint16_t *a[NR];')
    replace('    s->recv_mask[slot] = 0; s->sequence++;',
            '    if (tr) tr->reduced_ns = trace_now_ns();\n    memset(s->trace_recv_ns[slot], 0, sizeof(s->trace_recv_ns[slot]));\n    s->recv_mask[slot] = 0; s->sequence++;\n    trace_publish(s, tr);')
    replace('    free(s);', '    if (s->trace) munmap(s->trace, sizeof(*s->trace));\n    free(s);')
    replace('    int setup_result = setup(s);',
            '    int setup_result = trace_initialize(s);\n    if (!setup_result) setup_result = setup(s);')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(encoding='utf-8')), encoding='utf-8')
