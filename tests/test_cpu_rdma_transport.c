/* CPU-only C-ABI guards. Link with --wrap=ibv_get_device_list: no NIC opened. */
#define _POSIX_C_SOURCE 200809L
#include "cpu_rdma_transport.h"
#include <assert.h>
#include <errno.h>
#include <fenv.h>
#include <immintrin.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned queries;
struct ibv_device **__wrap_ibv_get_device_list(int *count) {
    ++queries; if (count) *count = 0; errno = ENODEV; return NULL;
}
int main(void) {
    char error[256];
    unsigned control = _mm_getcsr() & ~0x3fu;
    assert(cpu_rdma_numerics_selftest(error, sizeof(error)) == CPU_RDMA_OK);
    assert((_mm_getcsr() & ~0x3fu) == control);
    assert(fegetround() == FE_TONEAREST);
    assert(!queries);
    assert(fesetround(FE_DOWNWARD) == 0);
    assert(cpu_rdma_numerics_selftest(error, sizeof(error)) == CPU_RDMA_UNSUPPORTED);
    assert(fegetround() == FE_DOWNWARD);
    assert(fesetround(FE_TONEAREST) == 0);
    unsigned saved = _mm_getcsr();
    _mm_setcsr(saved | 0x8000u | 0x40u);
    assert(cpu_rdma_numerics_selftest(error, sizeof(error)) == CPU_RDMA_UNSUPPORTED);
    assert((_mm_getcsr() & (0x8000u | 0x40u)) == (0x8000u | 0x40u));
    _mm_setcsr(saved);
    void *arena = NULL;
    assert(posix_memalign(&arena, 64, CPU_RDMA_ARENA_BYTES) == 0);
    memset(arena, 0xa5, CPU_RDMA_ARENA_BYTES);
    cpu_rdma_params p = {.abi_version=1, .rank=0, .world_size=4,
        .run_id="0123456789abcdef0123456789abcdef", .device="mock-device",
        .master_ipv4="192.168.100.1", .master_port=29872, .gid_index=1, .timeout_ms=1000};
    cpu_rdma_ctx *ctx = NULL;
    assert(cpu_rdma_create(NULL, arena, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    assert(cpu_rdma_create(&p, NULL, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    assert(cpu_rdma_create(&p, (char*)arena+1, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    assert(cpu_rdma_create(&p, arena, CPU_RDMA_ARENA_BYTES-1, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    p.rank=4;
    assert(cpu_rdma_create(&p, arena, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    p.rank=0; p.run_id="not-a-run-id";
    assert(cpu_rdma_create(&p, arena, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    p.run_id="0123456789abcdef0123456789abcdef"; p.world_size=2;
    assert(cpu_rdma_create(&p, arena, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    p.world_size=4; p.timeout_ms=1;
    assert(cpu_rdma_create(&p, arena, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_INVALID);
    p.timeout_ms=1000;
    assert(!queries && !ctx);
    assert(cpu_rdma_create(&p, arena, CPU_RDMA_ARENA_BYTES, &ctx, error, sizeof(error)) == CPU_RDMA_SYSTEM);
    assert(queries == 1 && ctx == NULL);
    for (size_t i=CPU_RDMA_INPUT_OFFSET; i<CPU_RDMA_ARENA_BYTES; ++i)
        assert(((unsigned char*)arena)[i] == 0xa5);
    assert(cpu_rdma_run(NULL, NULL, NULL, CPU_RDMA_VALUES, CPU_RDMA_FP32_THEN_BF16) == CPU_RDMA_INVALID);
    assert(cpu_rdma_destroy(NULL) == CPU_RDMA_OK);
    free(arena);
    puts("{\"event\":\"cpu_transport_unit\",\"numeric_parity\":true,\"fp_state_guard\":true,\"invalid_args_guard\":true,\"borrowed_arena_survives_create_failure\":true,\"device_queries_mocked\":1,\"network_or_gpu_access\":false}");
    return 0;
}
