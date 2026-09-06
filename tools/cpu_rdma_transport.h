#ifndef CPU_RDMA_TRANSPORT_H
#define CPU_RDMA_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CPU_RDMA_ABI_VERSION 1u
#define CPU_RDMA_WORLD_SIZE 4u
#define CPU_RDMA_VALUES 10240u
#define CPU_RDMA_BYTES (CPU_RDMA_VALUES * 2u)
#define CPU_RDMA_ARENA_BYTES (CPU_RDMA_BYTES * 10u)
#define CPU_RDMA_INPUT_OFFSET (CPU_RDMA_BYTES * 8u)
#define CPU_RDMA_OUTPUT_OFFSET (CPU_RDMA_BYTES * 9u)

typedef struct cpu_rdma_ctx cpu_rdma_ctx;
typedef enum cpu_rdma_mode {
    CPU_RDMA_FP32_THEN_BF16 = 0,
    CPU_RDMA_BF16_EACH_ADD = 1
} cpu_rdma_mode;
typedef enum cpu_rdma_status {
    CPU_RDMA_OK = 0,
    CPU_RDMA_INVALID = -1,
    CPU_RDMA_SYSTEM = -2,
    CPU_RDMA_TIMEOUT = -3,
    CPU_RDMA_POISONED = -4,
    CPU_RDMA_UNSUPPORTED = -5
} cpu_rdma_status;

typedef struct cpu_rdma_params {
    uint32_t abi_version;
    uint32_t rank;
    uint32_t world_size;
    const char *run_id;       /* Exactly 32 lowercase hexadecimal characters. */
    const char *device;       /* Explicit active verbs HCA, <64 characters. */
    const char *master_ipv4;  /* Numeric IPv4 address of rank 0. */
    uint32_t master_port;     /* 1024..65535, bootstrap TCP only. */
    uint32_t gid_index;       /* 0..255, verbs port is fixed at 1. */
    uint32_t timeout_ms;      /* 100..120000, renewed for create and each run. */
} cpu_rdma_params;

/* Serialized single-caller API. Build this implementation as C, not C++.
 * arena is borrowed, >=64-byte aligned, and exactly CPU_RDMA_ARENA_BYTES.
 * Layout: eight alternating receive slots, stable input slot, stable output.
 * The entire caller-provided host allocation is registered once with verbs.
 * The caller must make GPU writes visible before run and host output visible
 * before GPU consumption. This module performs no HIP/GPU synchronization.
 *
 * create success transfers no allocation ownership. Failure normally leaves
 * *out NULL; non-NULL on failure means cleanup was incomplete: retain arena
 * and retry destroy or terminate the owning test process before freeing it.
 * No global signal handlers, alarm, process exit, or serving actions occur.
 */
int cpu_rdma_create(const cpu_rdma_params *params, void *arena, size_t arena_bytes,
                    cpu_rdma_ctx **out, char *error_buffer, size_t error_bytes);

/* Exact pointers arena+INPUT_OFFSET and arena+OUTPUT_OFFSET; values=10240.
 * No allocation, MR registration, TCP barrier, output logging or HIP calls.
 * All three send completions precede return, permitting input reuse afterward.
 * Error poisons the context: do not retry a failed collective on that context.
 * Successful output uses the tested fixed-rank AVX2/scalar numerical primitive.
 */
int cpu_rdma_run(cpu_rdma_ctx *ctx, const void *input_host, void *output_host,
                 size_t values, cpu_rdma_mode mode);

/* destroy(NULL) succeeds. Success frees ctx, never the borrowed arena.
 * Failure retains ctx/resources and requires arena lifetime to continue.
 * Native driver calls cannot be interrupted by the software poll deadline;
 * the owning standalone test must retain an external hard process timeout.
 */
int cpu_rdma_destroy(cpu_rdma_ctx *ctx);
const char *cpu_rdma_last_error(const cpu_rdma_ctx *ctx);

/* CPU-only, no device/network access, heap allocation or control-state changes.
 * Verifies current thread RNE + disabled FTZ/DAZ, then scalar/AVX2 bit parity
 * across rotating banks and exceptional/raw BF16 patterns. May set ordinary
 * FP exception status flags while exercising NaNs; it does not change masks,
 * rounding mode, FTZ or DAZ. Suitable once in the native callback thread.
 */
int cpu_rdma_numerics_selftest(char *error_buffer, size_t error_bytes);

#ifdef __cplusplus
}
#endif
#endif
