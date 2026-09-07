/* CPU-only serialized input/expected fixture test. No verbs context is opened. */
#include "cpu_rdma_transport.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fenv.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 3;
    uint16_t input[4][CPU_RDMA_VALUES], output[CPU_RDMA_VALUES], expected[CPU_RDMA_VALUES];
    const uint16_t *rows[4] = {input[0], input[1], input[2], input[3]};
    unsigned cases = 0;
    for (;;) {
        size_t size = fread(input, 1, sizeof(input), f);
        if (!size && feof(f)) break;
        if (size != sizeof(input) || fread(expected, 1, sizeof(expected), f) != sizeof(expected)) return 4;
        memset(output, 0xa5, sizeof(output));
        int rc = cpu_rdma_reduce_host(rows, output, CPU_RDMA_VALUES, CPU_RDMA_RCCL_RING4_BF16);
        if (rc) { fprintf(stderr, "arithmetic error case%u rc%d\n", cases, rc); return 5; }
        for (size_t i=0; i<CPU_RDMA_VALUES; ++i) if (output[i] != expected[i]) {
            fprintf(stderr, "mismatch case%u index%zu expected%04x got%04x\n", cases,i,expected[i],output[i]);
            return 6;
        }
        ++cases;
    }
    fclose(f);
    if (cases != 64) return 7;
    if (cpu_rdma_reduce_host(rows, output, CPU_RDMA_VALUES-1, CPU_RDMA_RCCL_RING4_BF16) != CPU_RDMA_INVALID) return 8;
    if (cpu_rdma_reduce_host(rows, output, CPU_RDMA_VALUES, (cpu_rdma_mode)3) != CPU_RDMA_INVALID) return 9;
    if (cpu_rdma_reduce_host(rows, input[0]+1, CPU_RDMA_VALUES, CPU_RDMA_RCCL_RING4_BF16) != CPU_RDMA_INVALID) return 10;
    if (fesetround(FE_DOWNWARD)) return 11;
    if (cpu_rdma_reduce_host(rows, output, CPU_RDMA_VALUES, CPU_RDMA_RCCL_RING4_BF16) != CPU_RDMA_UNSUPPORTED) return 12;
    if (fesetround(FE_TONEAREST)) return 13;
    printf("{\"cases\":%u,\"values_checked\":%u,\"exact\":true,\"guard_checks\":true,\"gpu_or_network_calls\":false}\n", cases,cases*CPU_RDMA_VALUES);
    return 0;
}
