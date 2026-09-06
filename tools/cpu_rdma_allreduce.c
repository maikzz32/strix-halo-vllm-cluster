/* Standalone host-memory RC allgather + local BF16 reduction.
 * No GPU/MPI/UCX dependency. Build CPU reference without verbs:
 * cc -O3 -std=c11 -Wall -Wextra -Werror -DCPU_ONLY ... -lm
 * Full build: cc -O3 -std=c11 -Wall -Wextra -Werror ... -libverbs -lm
 * Network execution requires an explicit benchmark window; --self-test is CPU only.
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { NR = 4, N = 10240, PERIOD = 32, BYTES = N * 2 };
static float from_bf16(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f;
}
static uint16_t to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u)
        return (u & 0x7fffffu) ? (uint16_t)((u >> 16 & 0x8000u) | 0x7fc0u) : (uint16_t)(u >> 16);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint16_t input_bits(unsigned rank, unsigned it, unsigned i) {
    static const uint16_t cancellation[4] = {0x4380, 0x3f80, 0xc380, 0x3f80};
    static const uint16_t small[4] = {0x3f80, 0x3b80, 0xbf80, 0x3b80};
    if (i % 16 == 0) return cancellation[rank];
    if (i % 16 == 1) return small[rank];
    uint32_t h = i * 0x9e3779b1u ^ ((it % PERIOD) + 1) * 0x85ebca6bu ^ (rank + 1) * 0xc2b2ae35u;
    h ^= h >> 16;
    return (uint16_t)(((h >> 15) & 0x8000u) | ((123 + ((h >> 24) % 9)) << 7) | (h & 127));
}
/* One compiled reference also fixes NaN operand/sign selection for SIMD fallback. */
__attribute__((noinline, noipa))
static uint16_t reduce_four(const uint16_t a[4], int round_each) {
    float f = from_bf16(a[0]);
    for (int r = 1; r < NR; ++r) {
        f = f + from_bf16(a[r]);
        if (round_each) f = from_bf16(to_bf16(f));
    }
    return to_bf16(f);
}
#include "cpu_rdma_reduce_avx2.h"
static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9;
}
__attribute__((noinline, optimize("no-tree-vectorize")))
static void reduce_scalar(const uint16_t *a[4], uint16_t *out, size_t count, int mode) {
    for (size_t i = 0; i < count; ++i) {
        uint16_t v[4] = {a[0][i], a[1][i], a[2][i], a[3][i]};
        out[i] = reduce_four(v, mode);
    }
}
static int scalar_vector_check(const uint16_t *a[4], size_t count, uint16_t *scalar, uint16_t *vector) {
    for (int mode = 0; mode < 2; ++mode) {
        reduce_scalar(a, scalar, count, mode); reduce_avx2(a, vector, count, mode);
        if (memcmp(scalar, vector, count * sizeof(uint16_t))) {
            for (size_t i = 0; i < count; ++i) if (scalar[i] != vector[i]) {
                fprintf(stderr, "SIMD mismatch mode%d index%zu inputs=%04x,%04x,%04x,%04x scalar=%04x vector=%04x\n",
                    mode, i, a[0][i], a[1][i], a[2][i], a[3][i], scalar[i], vector[i]); break;
            }
            return -1;
        }
    }
    return 0;
}
static int sum_self_test(void) {
    enum { EDGE_N = 65536, EDGE_CASES = 8, REPEATS = 256 };
    uint16_t *data = malloc(4 * EDGE_N * sizeof(uint16_t));
    uint16_t *scalar = malloc(EDGE_N * sizeof(uint16_t)), *vector = malloc(EDGE_N * sizeof(uint16_t));
    if (!data || !scalar || !vector) { free(data); free(scalar); free(vector); return 2; }
    const uint16_t *a[4]; for (int r = 0; r < 4; ++r) a[r] = data + r * EDGE_N;
    int rc = 2;
    for (unsigned bank = 0; bank < PERIOD; ++bank) {
        for (int r = 0; r < 4; ++r) for (unsigned i = 0; i < N; ++i) data[r * EDGE_N + i] = input_bits((unsigned)r, bank, i);
        if (scalar_vector_check(a, N, scalar, vector)) goto done;
    }
    /* Exhaust every input bit pattern in every rank position, plus mixed signs,
       infinities/NaNs, subnormals, cancellations and finite overflow sequences. */
    for (unsigned pattern = 0; pattern < EDGE_CASES; ++pattern) {
        for (unsigned i = 0; i < EDGE_N; ++i) for (unsigned r = 0; r < 4; ++r) {
            uint16_t v;
            if (pattern < 4) v = r == pattern ? (uint16_t)i : (r & 1u) ? 0xbf80 : 0x3f80;
            else if (pattern == 4) v = (uint16_t)(i + r * 16381u);
            else if (pattern == 5) v = (uint16_t)i;
            else if (pattern == 6) v = (r == 0 || r == 2) ? (uint16_t)(i ^ (r == 2 ? 0x8000u : 0)) : 0x3f80;
            else v = (uint16_t)((i * 40503u + r * 49157u) ^ ((i + r) >> 3));
            data[r * EDGE_N + i] = v;
        }
        if (scalar_vector_check(a, EDGE_N, scalar, vector)) goto done;
        /* Unaligned pointers and non-vector-width tails use the same semantics. */
        const uint16_t *tail[4] = {a[0] + 1, a[1] + 1, a[2] + 1, a[3] + 1};
        if (scalar_vector_check(tail, 1031, scalar, vector)) goto done;
    }
    printf("{\"event\":\"simd_self_test\",\"avx2_available\":%s,\"banks\":%d,\"edge_cases\":%d,\"edge_count\":%d,\"bitexact\":true}\n",
        reduction_avx2_available() ? "true" : "false", PERIOD, EDGE_CASES, EDGE_N);
    for (int r = 0; r < 4; ++r) for (unsigned i = 0; i < N; ++i) data[r * EDGE_N + i] = input_bits((unsigned)r, 0, i);
    for (int mode = 0; mode < 2; ++mode) for (int impl = 0; impl < 2; ++impl) {
        double elapsed = 0; volatile uint32_t checksum = 0;
        for (int it = 0; it < REPEATS + 16; ++it) {
            double begin = now_sec();
            if (impl) reduce_avx2(a, vector, N, mode); else reduce_scalar(a, vector, N, mode);
            double t = now_sec() - begin;
            checksum ^= vector[(unsigned)it % N];
            if (it >= 16) elapsed += t;
        }
        printf("{\"event\":\"sum_timing\",\"mode\":\"%s\",\"implementation\":\"%s\",\"mean_us\":%.6f,\"count\":%d,\"iterations\":%d,\"checksum\":%u}\n",
            mode ? "bf16_each_add" : "fp32_then_bf16", impl ? "avx2" : "scalar", elapsed * 1e6 / REPEATS, N, REPEATS, checksum);
    }
    rc = 0;
done:
    free(data); free(scalar); free(vector); return rc;
}
static uint64_t hash_add(uint64_t h, uint16_t v) {
    h = (h ^ (v & 255u)) * UINT64_C(1099511628211);
    return (h ^ (v >> 8)) * UINT64_C(1099511628211);
}
static int self_test(void) {
    uint64_t hi = UINT64_C(14695981039346656037), hf = hi, hb = hi;
    unsigned different = 0; double max_diff = 0, max_err_fp32 = 0, max_err_bf16 = 0;
    for (unsigned it = 0; it < PERIOD; ++it) for (unsigned i = 0; i < N; ++i) {
        uint16_t a[NR]; double exact = 0;
        for (unsigned r = 0; r < NR; ++r) {
            a[r] = input_bits(r, it, i); hi = hash_add(hi, a[r]);
            if (!isfinite(from_bf16(a[r])) || from_bf16(a[r]) == 0) return 2;
            exact += (double)from_bf16(a[r]);
        }
        uint16_t f = reduce_four(a, 0), b = reduce_four(a, 1);
        hf = hash_add(hf, f); hb = hash_add(hb, b); different += f != b;
        max_diff = fmax(max_diff, fabs((double)from_bf16(f) - from_bf16(b)));
        max_err_fp32 = fmax(max_err_fp32, fabs((double)from_bf16(f) - exact));
        max_err_bf16 = fmax(max_err_bf16, fabs((double)from_bf16(b) - exact));
    }
    if (!different || to_bf16(1.00390625f) != 0x3f80 || to_bf16(1.01171875f) != 0x3f82) return 2;
    printf("{\"event\":\"self_test\",\"inputs\":\"%016" PRIx64 "\",\"fp32\":\"%016" PRIx64
           "\",\"bf16\":\"%016" PRIx64 "\",\"elements\":%u,\"different_outputs\":%u,"
           "\"max_mode_difference\":%.9g,\"max_fp32_abs_error\":%.9g,\"max_bf16_abs_error\":%.9g}\n",
           hi, hf, hb, PERIOD * N, different, max_diff, max_err_fp32, max_err_bf16);
    return 0;
}

#ifndef CPU_ONLY
#include <arpa/inet.h>
#include <endian.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <poll.h>
#include <sys/socket.h>

static volatile sig_atomic_t interrupted;
static double deadline;
static void on_signal(int sig) {
    (void)sig;
    if (interrupted) _exit(124);
    interrupted = 1; alarm(3);
}
static int expired(void) { return interrupted || now_sec() >= deadline; }
static int wait_fd(int fd, short events) {
    while (!expired()) {
        struct pollfd p = {.fd = fd, .events = events};
        int n = poll(&p, 1, 100);
        if (n > 0) return (p.revents & events) ? 0 : -1;
        if (n < 0 && errno != EINTR) return -1;
    }
    return -1;
}
static int transfer(int fd, void *data, size_t bytes, int writing) {
    char *p = data;
    while (bytes) {
        if (wait_fd(fd, writing ? POLLOUT : POLLIN)) return -1;
        ssize_t n = writing ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n <= 0) return -1;
        p += n; bytes -= (size_t)n;
    }
    return 0;
}
static int nonblocking(int fd) { return fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK); }
struct endpoint {
    uint32_t magic, rank, count, mtu, rkey, qpn[NR], psn[NR];
    uint64_t address;
    uint8_t gid[16];
    char run_id[33];
    uint8_t pad[7];
};
struct state {
    int rank, gid_index, port, seconds, iterations, warmup, mode, use_avx2;
    const char *device, *master, *run_id;
    int listener, fd[NR];
    struct ibv_device **devices;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_mr *mr;
    struct ibv_qp *qp[NR];
    uint16_t *memory, *banks, *output, *expected[2];
    struct endpoint endpoints[NR];
    uint32_t recv_next[NR], recv_mask[2], sequence;
};
static int valid_endpoint(const struct endpoint *e, int rank, const char *run_id) {
    return ntohl(e->magic) == 0x43524431u && ntohl(e->rank) == (unsigned)rank &&
        ntohl(e->count) == N && !memcmp(e->run_id, run_id, 33) &&
        ntohl(e->mtu) >= IBV_MTU_4096;
}
static int bootstrap(struct state *s) {
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)s->port)};
    if (inet_pton(AF_INET, s->master, &addr.sin_addr) != 1) return -1;
    if (s->rank == 0) {
        s->listener = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        if (s->listener < 0 || setsockopt(s->listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) ||
            nonblocking(s->listener) || bind(s->listener, (void *)&addr, sizeof(addr)) || listen(s->listener, NR)) return -1;
        for (int n = 1; n < NR; ++n) {
            if (wait_fd(s->listener, POLLIN)) return -1;
            int fd = accept(s->listener, NULL, NULL);
            if (fd < 0) return -1;
            struct endpoint e;
            if (nonblocking(fd) || transfer(fd, &e, sizeof(e), 0)) { close(fd); return -1; }
            int rank = (int)ntohl(e.rank);
            if (rank < 1 || rank >= NR || s->fd[rank] >= 0 || !valid_endpoint(&e, rank, s->run_id)) { close(fd); return -1; }
            s->fd[rank] = fd; s->endpoints[rank] = e;
        }
        for (int r = 1; r < NR; ++r) if (transfer(s->fd[r], s->endpoints, sizeof(s->endpoints), 1)) return -1;
        close(s->listener); s->listener = -1;
    } else {
        while (!expired()) {
            int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) return -1;
            if (nonblocking(fd)) { close(fd); return -1; }
            int rc = connect(fd, (void *)&addr, sizeof(addr));
            if (rc && errno == EINPROGRESS && !wait_fd(fd, POLLOUT)) {
                int error = 0; socklen_t len = sizeof(error);
                if (!getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len) && !error) rc = 0;
            }
            if (!rc) { s->fd[0] = fd; break; }
            close(fd); struct timespec pause = {0, 50000000}; nanosleep(&pause, NULL);
        }
        if (s->fd[0] < 0 || transfer(s->fd[0], &s->endpoints[s->rank], sizeof(struct endpoint), 1) ||
            transfer(s->fd[0], s->endpoints, sizeof(s->endpoints), 0)) return -1;
    }
    for (int r = 0; r < NR; ++r) if (!valid_endpoint(&s->endpoints[r], r, s->run_id)) return -1;
    return 0;
}
static int barrier(struct state *s, uint32_t phase) {
    uint32_t value = htonl(phase), got;
    if (s->rank == 0) {
        for (int r = 1; r < NR; ++r) if (transfer(s->fd[r], &got, 4, 0) || got != value) return -1;
        for (int r = 1; r < NR; ++r) if (transfer(s->fd[r], &value, 4, 1)) return -1;
    } else if (transfer(s->fd[0], &value, 4, 1) || transfer(s->fd[0], &got, 4, 0) || got != value) return -1;
    return 0;
}
static int post_receive(struct state *s, int peer) {
    struct ibv_recv_wr wr = {.wr_id = (uint64_t)peer}, *bad = NULL;
    return ibv_post_recv(s->qp[peer], &wr, &bad);
}
static int setup(struct state *s) {
    size_t bytes = (2 * NR + PERIOD) * (size_t)BYTES;
    if (posix_memalign((void **)&s->memory, 4096, bytes)) return -1;
    memset(s->memory, 0, bytes);
    s->banks = s->memory + 2 * NR * N;
    s->output = malloc(BYTES);
    for (int m = 0; m < 2; ++m) s->expected[m] = malloc(PERIOD * BYTES);
    if (!s->output || !s->expected[0] || !s->expected[1]) return -1;
    for (unsigned it = 0; it < PERIOD; ++it) for (unsigned i = 0; i < N; ++i) {
        uint16_t a[NR];
        for (unsigned r = 0; r < NR; ++r) a[r] = input_bits(r, it, i);
        s->banks[it * N + i] = a[s->rank];
        for (int m = 0; m < 2; ++m) s->expected[m][it * N + i] = reduce_four(a, m);
    }
    s->devices = ibv_get_device_list(NULL);
    if (!s->devices) return -1;
    for (int i = 0; s->devices[i]; ++i)
        if (!strcmp(ibv_get_device_name(s->devices[i]), s->device)) { s->ctx = ibv_open_device(s->devices[i]); break; }
    if (!s->ctx) return -1;
    struct ibv_port_attr port;
    union ibv_gid gid;
    if (ibv_query_port(s->ctx, 1, &port) || port.state != IBV_PORT_ACTIVE || port.active_mtu < IBV_MTU_4096 ||
        ibv_query_gid(s->ctx, 1, s->gid_index, &gid)) return -1;
    s->pd = ibv_alloc_pd(s->ctx);
    if (!s->pd) return -1;
    s->cq = ibv_create_cq(s->ctx, 128, NULL, NULL, 0);
    if (!s->cq) return -1;
    s->mr = ibv_reg_mr(s->pd, s->memory, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!s->mr) return -1;
    struct endpoint *e = &s->endpoints[s->rank];
    e->magic = htonl(0x43524431u); e->rank = htonl((uint32_t)s->rank); e->count = htonl(N);
    e->mtu = htonl((uint32_t)port.active_mtu); e->address = htobe64((uint64_t)(uintptr_t)s->memory); e->rkey = htonl(s->mr->rkey);
    memcpy(e->gid, &gid, 16); memcpy(e->run_id, s->run_id, 33);
    for (int r = 0; r < NR; ++r) if (r != s->rank) {
        struct ibv_qp_init_attr init = {.send_cq = s->cq, .recv_cq = s->cq, .qp_type = IBV_QPT_RC,
            .cap = {.max_send_wr = 16, .max_recv_wr = 16, .max_send_sge = 1, .max_recv_sge = 1}};
        s->qp[r] = ibv_create_qp(s->pd, &init);
        if (!s->qp[r]) return -1;
        struct ibv_qp_attr attr = {.qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = 1,
            .qp_access_flags = IBV_ACCESS_REMOTE_WRITE};
        if (ibv_modify_qp(s->qp[r], &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return -1;
        e->qpn[r] = htonl(s->qp[r]->qp_num);
        e->psn[r] = htonl((s->qp[r]->qp_num * 2654435761u ^ (uint32_t)getpid()) & 0xffffffu);
        for (int j = 0; j < 4; ++j) if (post_receive(s, r)) return -1;
    }
    if (bootstrap(s)) return -1;
    for (int r = 0; r < NR; ++r) if (r != s->rank) {
        struct endpoint *p = &s->endpoints[r];
        struct ibv_qp_attr a = {.qp_state = IBV_QPS_RTR, .path_mtu = IBV_MTU_4096,
            .dest_qp_num = ntohl(p->qpn[s->rank]), .rq_psn = ntohl(p->psn[s->rank]),
            .max_dest_rd_atomic = 1, .min_rnr_timer = 12,
            .ah_attr = {.is_global = 1, .port_num = 1, .grh = {.sgid_index = (uint8_t)s->gid_index, .hop_limit = 64}}};
        memcpy(&a.ah_attr.grh.dgid, p->gid, 16);
        if (ibv_modify_qp(s->qp[r], &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) return -1;
        memset(&a, 0, sizeof(a)); a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 3; a.rnr_retry = 3;
        a.sq_psn = ntohl(e->psn[r]); a.max_rd_atomic = 1;
        if (ibv_modify_qp(s->qp[r], &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) return -1;
    }
    return barrier(s, 1);
}
static int collective(struct state *s, int mode, double *elapsed, double *transport, double *sum) {
    uint32_t seq = s->sequence, slot = seq & 1u, full = 15u ^ (1u << s->rank), sent = 0;
    uint16_t *input = s->banks + (seq % PERIOD) * N;
    double begin = now_sec();
    for (int r = 0; r < NR; ++r) if (r != s->rank) {
        struct ibv_sge sg = {.addr = (uint64_t)(uintptr_t)input, .length = BYTES, .lkey = s->mr->lkey};
        struct ibv_send_wr wr = {.wr_id = (uint64_t)(0x100 | r), .sg_list = &sg, .num_sge = 1,
            .opcode = IBV_WR_RDMA_WRITE_WITH_IMM, .send_flags = IBV_SEND_SIGNALED, .imm_data = htonl(seq),
            .wr.rdma = {.remote_addr = be64toh(s->endpoints[r].address) + ((slot * NR + s->rank) * (uint64_t)BYTES),
                        .rkey = ntohl(s->endpoints[r].rkey)}}, *bad = NULL;
        if (ibv_post_send(s->qp[r], &wr, &bad)) return -1;
    }
    unsigned polls = 0;
    while (sent != full || s->recv_mask[slot] != full) {
        struct ibv_wc wc[16]; int count = ibv_poll_cq(s->cq, 16, wc);
        if (count < 0) return -1;
        if ((++polls & 1023u) == 0 && expired()) return -1;
        for (int j = 0; j < count; ++j) {
            struct ibv_wc *w = &wc[j]; int r = (int)(w->wr_id & 255u);
            if (w->status != IBV_WC_SUCCESS) {
                fprintf(stderr, "rank %d seq %u completion error: %s vendor=%u peer=%d\n", s->rank, seq,
                        ibv_wc_status_str(w->status), w->vendor_err, r); return -1;
            }
            if (r < 0 || r >= NR || r == s->rank) return -1;
            if (w->wr_id & 0x100u) {
                if (w->opcode != IBV_WC_RDMA_WRITE || (sent & (1u << r))) return -1;
                sent |= 1u << r;
            } else {
                uint32_t got = ntohl(w->imm_data);
                if (w->opcode != IBV_WC_RECV_RDMA_WITH_IMM || !(w->wc_flags & IBV_WC_WITH_IMM) ||
                    got != s->recv_next[r] || got < seq || got > seq + 1 ||
                    (s->recv_mask[got & 1u] & (1u << r))) return -1;
                s->recv_next[r]++; s->recv_mask[got & 1u] |= 1u << r;
                if (post_receive(s, r)) return -1;
            }
        }
    }
    /* Two slots suffice: a peer cannot send seq+2 before receiving our seq+1,
       which we send only after completing this read/reduction/check. */
    double completed = now_sec();
    const uint16_t *a[NR];
    for (int r = 0; r < NR; ++r) a[r] = r == s->rank ? input : s->memory + (slot * NR + r) * N;
    if (s->use_avx2) reduce_avx2(a, s->output, N, mode);
    else reduce_scalar(a, s->output, N, mode);
    double reduced = now_sec();
    *transport = (completed - begin) * 1e6;
    *sum = (reduced - completed) * 1e6;
    *elapsed = (reduced - begin) * 1e6;
    if (memcmp(s->output, s->expected[mode] + (seq % PERIOD) * N, BYTES)) {
        fprintf(stderr, "rank %d seq %u numerical mismatch\n", s->rank, seq); return -1;
    }
    s->recv_mask[slot] = 0; s->sequence++;
    return 0;
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
static int benchmark(struct state *s) {
    double *times = calloc((size_t)s->iterations, sizeof(double));
    if (!times) return -1;
    int result = -1;
    for (int mode = 0; mode < 2; ++mode) if (s->mode == 2 || mode == s->mode) {
        if (barrier(s, (uint32_t)(10 + mode))) goto done;
        for (int i = 0; i < s->warmup; ++i) { double t, net, sum; if (collective(s, mode, &t, &net, &sum)) goto done; }
        double start = now_sec();
        double network_total = 0, sum_total = 0;
        for (int i = 0; i < s->iterations; ++i) {
            double net, sum;
            if (collective(s, mode, &times[i], &net, &sum)) goto done;
            network_total += net; sum_total += sum;
        }
        double wall = (now_sec() - start) * 1e6 / s->iterations;
        qsort(times, (size_t)s->iterations, sizeof(double), compare_double);
        printf("{\"event\":\"result\",\"run_id\":\"%s\",\"rank\":%d,\"mode\":\"%s\","
               "\"algorithm\":\"allgather_three_parallel_writes_then_local_sum\",\"values\":%d,\"payload_bytes\":%d,"
               "\"tx_payload_bytes_per_rank\":%d,\"iterations\":%d,\"warmup\":%d,\"correct\":true,"
               "\"min_us\":%.6f,\"median_us\":%.6f,\"p95_us\":%.6f,\"max_us\":%.6f,\"wall_us_including_check\":%.6f,"
               "\"implementation\":\"%s\",\"transport_mean_us\":%.6f,\"sum_mean_us\":%.6f}\n",
               s->run_id, s->rank, mode ? "bf16_each_add" : "fp32_then_bf16", N, BYTES, 3 * BYTES,
               s->iterations, s->warmup, times[0], times[s->iterations / 2], times[(s->iterations * 95) / 100],
               times[s->iterations - 1], wall, s->use_avx2 ? "avx2" : "scalar", network_total / s->iterations, sum_total / s->iterations);
        fflush(stdout);
    }
    result = barrier(s, 99);
done:
    free(times); return result;
}
static int cleanup(struct state *s) {
    int error = 0;
    for (int r = 0; r < NR; ++r) if (s->qp[r] && ibv_destroy_qp(s->qp[r])) error = 1;
    if (s->mr && ibv_dereg_mr(s->mr)) error = 1;
    if (s->cq && ibv_destroy_cq(s->cq)) error = 1;
    if (s->pd && ibv_dealloc_pd(s->pd)) error = 1;
    if (s->ctx && ibv_close_device(s->ctx)) error = 1;
    if (s->devices) ibv_free_device_list(s->devices);
    for (int r = 0; r < NR; ++r) if (s->fd[r] >= 0) close(s->fd[r]);
    if (s->listener >= 0) close(s->listener);
    free(s->memory); free(s->output); free(s->expected[0]); free(s->expected[1]);
    return error;
}
static int parse_int(const char *v, int min, int max) {
    char *end; errno = 0; long x = strtol(v, &end, 10);
    return errno || !*v || *end || x < min || x > max ? -1 : (int)x;
}
static int network_main(int argc, char **argv) {
    struct state s = {.rank = -1, .gid_index = 1, .port = 29871, .seconds = 30, .iterations = 128,
                      .warmup = 16, .mode = 2, .master = "192.168.100.1", .listener = -1};
    for (int r = 0; r < NR; ++r) s.fd[r] = -1;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) return 2;
        const char *k = argv[i], *v = argv[i + 1];
        if (!strcmp(k, "--rank")) s.rank = parse_int(v, 0, 3);
        else if (!strcmp(k, "--device")) s.device = v;
        else if (!strcmp(k, "--run-id")) s.run_id = v;
        else if (!strcmp(k, "--master")) s.master = v;
        else if (!strcmp(k, "--port")) s.port = parse_int(v, 1024, 65535);
        else if (!strcmp(k, "--gid-index")) s.gid_index = parse_int(v, 0, 255);
        else if (!strcmp(k, "--deadline")) s.seconds = parse_int(v, 5, 120);
        else if (!strcmp(k, "--iterations")) s.iterations = parse_int(v, 8, 4096);
        else if (!strcmp(k, "--warmup")) s.warmup = parse_int(v, 1, 256);
        else if (!strcmp(k, "--mode")) s.mode = !strcmp(v, "fp32") ? 0 : !strcmp(v, "bf16") ? 1 : !strcmp(v, "both") ? 2 : -1;
        else if (!strcmp(k, "--implementation")) s.use_avx2 = !strcmp(v, "scalar") ? 0 : !strcmp(v, "avx2") ? 1 : -1;
        else return 2;
    }
    if (s.rank < 0 || !s.device || !s.run_id || strlen(s.run_id) != 32 ||
        strspn(s.run_id, "0123456789abcdef") != 32 || s.port < 0 || s.gid_index < 0 || s.seconds < 0 ||
        s.iterations < 0 || s.warmup < 0 || s.mode < 0 || s.use_avx2 < 0 ||
        (s.use_avx2 && !reduction_avx2_available())) return 2;
    signal(SIGTERM, on_signal); signal(SIGINT, on_signal); signal(SIGALRM, on_signal); signal(SIGPIPE, SIG_IGN);
    deadline = now_sec() + s.seconds; alarm((unsigned)s.seconds);
    printf("{\"event\":\"start\",\"run_id\":\"%s\",\"rank\":%d,\"pid\":%ld,\"device\":\"%s\",\"gid_index\":%d}\n",
           s.run_id, s.rank, (long)getpid(), s.device, s.gid_index); fflush(stdout);
    int rc = setup(&s);
    if (!rc) rc = benchmark(&s);
    if (rc) fprintf(stderr, "rank %d failed or timed out (errno=%d)\n", s.rank, errno);
    if (cleanup(&s)) rc = -1;
    alarm(0);
    if (rc) return interrupted ? 124 : 1;
    printf("{\"event\":\"complete\",\"run_id\":\"%s\",\"rank\":%d,\"teardown_ok\":true}\n", s.run_id, s.rank);
    return 0;
}
#endif
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) return self_test();
    if (argc == 2 && !strcmp(argv[1], "--sum-self-test")) return sum_self_test();
#ifdef CPU_ONLY
    fputs("CPU_ONLY build accepts --self-test only\n", stderr); return 2;
#else
    return network_main(argc, argv);
#endif
}
