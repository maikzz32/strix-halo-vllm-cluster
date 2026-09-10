/* CPU-only cost of the production ring4 reducer; no verbs context or HIP calls.
 * Build with cpu_rdma_transport.c, -O3 and without fast-math or LTO.
 * Hot/rotating malloc buffers are arithmetic estimates, not GPU-coherence timing. */
#define _POSIX_C_SOURCE 200809L
#include "cpu_rdma_transport.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static double seconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) abort();
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    enum { BANKS=32, LOOPS=2048, SAMPLES=5 };
    char message[512];
    if (cpu_rdma_numerics_selftest(message,sizeof(message))) {
        fprintf(stderr,"%s\n",message); return 1;
    }
    uint16_t *data=aligned_alloc(64,BANKS*5*CPU_RDMA_BYTES);
    if (!data) return 2;
    for (unsigned b=0;b<BANKS;b++) for(unsigned r=0;r<4;r++)
        for(unsigned i=0;i<CPU_RDMA_VALUES;i++) {
            uint32_t h=(i+1)*0x9e3779b1u^(b+1)*0x85ebca6bu^(r+1)*0xc2b2ae35u;
            h^=h>>16;
            data[(b*5+r)*CPU_RDMA_VALUES+i]=(uint16_t)(
                ((h>>15)&0x8000u)|((123+((h>>24)%9))<<7)|(h&127));
        }
    for (unsigned banks=1;banks<=BANKS;banks*=BANKS) {
        for(unsigned sample=0;sample<SAMPLES;sample++) {
            double start=seconds();
            for(unsigned it=0;it<LOOPS;it++) {
                unsigned b=it%banks;
                const uint16_t *a[4];
                for(unsigned r=0;r<4;r++)a[r]=data+(b*5+r)*CPU_RDMA_VALUES;
                if(cpu_rdma_reduce_host(a,data+(b*5+4)*CPU_RDMA_VALUES,
                    CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16)) return 3;
            }
            double elapsed=seconds()-start;
            uint64_t sum=0;
            for(unsigned b=0;b<banks;b++)for(unsigned i=0;i<CPU_RDMA_VALUES;i++)
                sum+=data[(b*5+4)*CPU_RDMA_VALUES+i];
            printf("{\"banks\":%u,\"sample\":%u,\"calls\":%u,\"mean_us\":%.9f,\"checksum\":%llu}\n",
                banks,sample,LOOPS,elapsed*1e6/LOOPS,(unsigned long long)sum);
        }
    }
    free(data);
    return 0;
}
