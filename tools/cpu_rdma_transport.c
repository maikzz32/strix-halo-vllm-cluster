/* Experimental CPU transport extracted from the validated standalone benchmark.
 * Compile as C11 in the SAME container/toolchain as the HIP wrapper.
 * No GPU, signal handler, process exit, or allocation in collective(). */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "cpu_rdma_transport.h"
#include <arpa/inet.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <infiniband/verbs.h>
#include <poll.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
enum { NR = 4, N = 10240, BYTES = 20480, PERIOD = 32 };
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
struct endpoint {
    uint32_t magic, rank, count, mtu, rkey, qpn[NR], psn[NR];
    uint64_t address;
    uint8_t gid[16];
    char run_id[33];
    uint8_t pad[7];
};

struct cpu_rdma_ctx {
    int rank, gid_index, port, listener, fd[NR], ready, poisoned;
    uint32_t timeout_ms;
    double deadline;
    char device[64], master[INET_ADDRSTRLEN], run_id[33], error[256];
    const char *stage;
    struct ibv_device **devices;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_mr *mr;
    struct ibv_qp *qp[NR];
    uint16_t *memory;
    struct endpoint endpoints[NR];
    uint32_t recv_next[NR], recv_mask[2], sequence;
};
static int expired(struct cpu_rdma_ctx *s) { return now_sec() >= s->deadline; }
static int fp_state_ok(void) {
    if (fegetround() != FE_TONEAREST) return 0;
#if defined(__x86_64__) || defined(__i386__)
    if (_mm_getcsr() & (0x6000u | 0x8000u | 0x0040u)) return 0;
#endif
    return 1;
}
int cpu_rdma_reduce_host(const uint16_t *inputs[4], uint16_t *output,
                         size_t values, cpu_rdma_mode mode) {
    if (!inputs || !output || values != N || mode < 0 || mode > 2)
        return CPU_RDMA_INVALID;
    for (unsigned r = 0; r < NR; ++r) {
        if (!inputs[r]) return CPU_RDMA_INVALID;
        uintptr_t a = (uintptr_t)inputs[r], b = (uintptr_t)output;
        if (a != b && (a > b ? a-b : b-a) < BYTES) return CPU_RDMA_INVALID;
    }
    if (!fp_state_ok()) return CPU_RDMA_UNSUPPORTED;
    if (mode != CPU_RDMA_RCCL_RING4_BF16) {
        reduce_avx2(inputs, output, values, (int)mode);
        return CPU_RDMA_OK;
    }
    /* Profile 2acfe604f80dbece21e2ee77e9accbc5952bf356833a1d5e2b722513a2c23f80.
     * Independently logged rings; source-derived 3072/3072/3072/1024 partition.
     * Owner's successor contributes first; owner contributes last. */
    static const unsigned rings[4][4] = {{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,1,3,2}};
    size_t offset = 0;
    for (unsigned channel = 0; channel < 4; ++channel) {
        size_t chunk = channel == 3 ? 256 : 768;
        for (unsigned owner = 0; owner < 4; ++owner) {
            const uint16_t *ordered[4];
            for (unsigned r = 0; r < 4; ++r)
                ordered[r] = inputs[rings[channel][(owner + 1 + r) % 4]] + offset;
            reduce_avx2(ordered, output + offset, chunk, 1);
            offset += chunk;
        }
    }
    return CPU_RDMA_OK;
}
static int wait_fd(struct cpu_rdma_ctx *s, int fd, short events) {
    while (!expired(s)) {
        struct pollfd p = {.fd = fd, .events = events};
        int left = (int)((s->deadline - now_sec()) * 1000.0);
        int n = poll(&p, 1, left < 1 ? 1 : left > 50 ? 50 : left);
        if (n > 0) return (p.revents & events) ? 0 : -1;
        if (n < 0 && errno != EINTR) return -1;
    }
    return -1;
}
static int transfer(struct cpu_rdma_ctx *s, int fd, void *data, size_t bytes, int writing) {
    char *p = data;
    while (bytes) {
        if (wait_fd(s, fd, writing ? POLLOUT : POLLIN)) return -1;
        ssize_t n = writing ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n <= 0) return -1;
        p += n; bytes -= (size_t)n;
    }
    return 0;
}
static int nonblocking(int fd) { return fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK); }
static int valid_endpoint(const struct endpoint *e, int rank, const char *run_id) {
    return ntohl(e->magic) == 0x43524431u && ntohl(e->rank) == (unsigned)rank &&
        ntohl(e->count) == N && !memcmp(e->run_id, run_id, 33) &&
        ntohl(e->mtu) >= IBV_MTU_4096;
}
static int bootstrap(struct cpu_rdma_ctx *s) {
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)s->port)};
    if (inet_pton(AF_INET, s->master, &addr.sin_addr) != 1) return -1;
    if (s->rank == 0) {
        s->listener = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        if (s->listener < 0 || setsockopt(s->listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) ||
            nonblocking(s->listener) || bind(s->listener, (void *)&addr, sizeof(addr)) || listen(s->listener, NR)) return -1;
        for (int n = 1; n < NR; ++n) {
            if (wait_fd(s, s->listener, POLLIN)) return -1;
            int fd = accept(s->listener, NULL, NULL);
            if (fd < 0) return -1;
            struct endpoint e;
            if (nonblocking(fd) || transfer(s, fd, &e, sizeof(e), 0)) { close(fd); return -1; }
            int rank = (int)ntohl(e.rank);
            if (rank < 1 || rank >= NR || s->fd[rank] >= 0 || !valid_endpoint(&e, rank, s->run_id)) { close(fd); return -1; }
            s->fd[rank] = fd; s->endpoints[rank] = e;
        }
        for (int r = 1; r < NR; ++r) if (transfer(s, s->fd[r], s->endpoints, sizeof(s->endpoints), 1)) return -1;
        close(s->listener); s->listener = -1;
    } else {
        while (!expired(s)) {
            int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) return -1;
            if (nonblocking(fd)) { close(fd); return -1; }
            int rc = connect(fd, (void *)&addr, sizeof(addr));
            if (rc && errno == EINPROGRESS && !wait_fd(s, fd, POLLOUT)) {
                int error = 0; socklen_t len = sizeof(error);
                if (!getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len) && !error) rc = 0;
            }
            if (!rc) { s->fd[0] = fd; break; }
            close(fd); struct timespec pause = {0, 50000000}; nanosleep(&pause, NULL);
        }
        if (s->fd[0] < 0 || transfer(s, s->fd[0], &s->endpoints[s->rank], sizeof(struct endpoint), 1) ||
            transfer(s, s->fd[0], s->endpoints, sizeof(s->endpoints), 0)) return -1;
    }
    for (int r = 0; r < NR; ++r) if (!valid_endpoint(&s->endpoints[r], r, s->run_id)) return -1;
    return 0;
}
static int barrier(struct cpu_rdma_ctx *s, uint32_t phase) {
    uint32_t value = htonl(phase), got;
    if (s->rank == 0) {
        for (int r = 1; r < NR; ++r) if (transfer(s, s->fd[r], &got, 4, 0) || got != value) return -1;
        for (int r = 1; r < NR; ++r) if (transfer(s, s->fd[r], &value, 4, 1)) return -1;
    } else if (transfer(s, s->fd[0], &value, 4, 1) || transfer(s, s->fd[0], &got, 4, 0) || got != value) return -1;
    return 0;
}
static int post_receive(struct cpu_rdma_ctx *s, int peer) {
    struct ibv_recv_wr wr = {.wr_id = (uint64_t)peer}, *bad = NULL;
    return ibv_post_recv(s->qp[peer], &wr, &bad);
}
static int setup(struct cpu_rdma_ctx *s) {
    s->stage = "open HCA";
    s->devices = ibv_get_device_list(NULL);
    if (!s->devices) return -1;
    for (int i = 0; s->devices[i]; ++i)
        if (!strcmp(ibv_get_device_name(s->devices[i]), s->device)) { s->ctx = ibv_open_device(s->devices[i]); break; }
    if (!s->ctx) return -1;
    s->stage = "active MTU/GID";
    struct ibv_port_attr port;
    union ibv_gid gid;
    if (ibv_query_port(s->ctx, 1, &port) || port.state != IBV_PORT_ACTIVE || port.active_mtu < IBV_MTU_4096 ||
        ibv_query_gid(s->ctx, 1, s->gid_index, &gid)) return -1;
    s->stage = "allocate PD";
    s->pd = ibv_alloc_pd(s->ctx);
    if (!s->pd) return -1;
    s->stage = "create CQ";
    s->cq = ibv_create_cq(s->ctx, 128, NULL, NULL, 0);
    if (!s->cq) return -1;
    s->stage = "register borrowed host arena";
    s->mr = ibv_reg_mr(s->pd, s->memory, CPU_RDMA_ARENA_BYTES, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!s->mr) return -1;
    s->stage = "create/init RC QPs";
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
    s->stage = "bootstrap/run-ID handshake";
    if (bootstrap(s)) return -1;
    s->stage = "connect RTR/RTS";
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
    s->stage = "initial ready barrier";
    return barrier(s, 1);
}
static int collective(struct cpu_rdma_ctx *s, const uint16_t *input, uint16_t *output, int mode) {
    uint32_t seq = s->sequence, slot = seq & 1u, full = 15u ^ (1u << s->rank), sent = 0;
    atomic_thread_fence(memory_order_release);
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
        if ((++polls & 1023u) == 0 && expired(s)) return -1;
        for (int j = 0; j < count; ++j) {
            struct ibv_wc *w = &wc[j]; int r = (int)(w->wr_id & 255u);
            if (w->status != IBV_WC_SUCCESS) {
                snprintf(s->error, sizeof(s->error), "rank %d seq %u completion %s vendor=%u peer=%d", s->rank, seq,
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
    atomic_thread_fence(memory_order_acquire);
    const uint16_t *a[NR];
    for (int r = 0; r < NR; ++r) a[r] = r == s->rank ? input : s->memory + (slot * NR + r) * N;
    if (cpu_rdma_reduce_host(a, output, N, (cpu_rdma_mode)mode) != CPU_RDMA_OK) return -1;
    atomic_thread_fence(memory_order_release);
    s->recv_mask[slot] = 0; s->sequence++;
    return 0;
}

static void copy_error(char *buffer, size_t bytes, const char *message) {
    if (buffer && bytes) snprintf(buffer, bytes, "%s", message);
}
const char *cpu_rdma_last_error(const cpu_rdma_ctx *s) {
    return s ? s->error : "no context";
}
int cpu_rdma_destroy(cpu_rdma_ctx *s) {
    if (!s) return CPU_RDMA_OK;
    s->ready = 0; s->poisoned = 1;
    /* Never deregister/free while a QP still exists. On failure retain the
       partially cleaned context so caller can retry without losing ownership. */
    for (int r = 0; r < NR; ++r) if (s->qp[r]) {
        if (ibv_destroy_qp(s->qp[r])) {
            snprintf(s->error, sizeof(s->error), "destroy QP peer%d failed errno%d; retain arena", r, errno);
            return CPU_RDMA_SYSTEM;
        }
        s->qp[r] = NULL;
    }
    if (s->mr) {
        if (ibv_dereg_mr(s->mr)) { copy_error(s->error, sizeof(s->error), "deregister MR failed; retain arena"); return CPU_RDMA_SYSTEM; }
        s->mr = NULL;
    }
    if (s->cq) {
        if (ibv_destroy_cq(s->cq)) { copy_error(s->error, sizeof(s->error), "destroy CQ failed"); return CPU_RDMA_SYSTEM; }
        s->cq = NULL;
    }
    if (s->pd) {
        if (ibv_dealloc_pd(s->pd)) { copy_error(s->error, sizeof(s->error), "deallocate PD failed"); return CPU_RDMA_SYSTEM; }
        s->pd = NULL;
    }
    if (s->ctx) {
        if (ibv_close_device(s->ctx)) { copy_error(s->error, sizeof(s->error), "close device failed"); return CPU_RDMA_SYSTEM; }
        s->ctx = NULL;
    }
    if (s->devices) ibv_free_device_list(s->devices);
    for (int r = 0; r < NR; ++r) if (s->fd[r] >= 0) close(s->fd[r]);
    if (s->listener >= 0) close(s->listener);
    /* memory is owned by the HIP harness and is intentionally never freed. */
    free(s);
    return CPU_RDMA_OK;
}
int cpu_rdma_create(const cpu_rdma_params *p, void *arena, size_t arena_bytes,
                    cpu_rdma_ctx **out, char *error_buffer, size_t error_bytes) {
    if (out) *out = NULL;
    copy_error(error_buffer, error_bytes, "");
    struct in_addr ip;
    if (!p || !out || !arena || (uintptr_t)arena % 64u || arena_bytes != CPU_RDMA_ARENA_BYTES ||
        (uintptr_t)arena > UINTPTR_MAX - CPU_RDMA_ARENA_BYTES || p->abi_version != CPU_RDMA_ABI_VERSION ||
        p->rank >= CPU_RDMA_WORLD_SIZE || p->world_size != CPU_RDMA_WORLD_SIZE ||
        !p->run_id || strnlen(p->run_id, 33) != 32 || strspn(p->run_id, "0123456789abcdef") != 32 ||
        !p->device || !*p->device || strnlen(p->device, 64) >= 64 ||
        !p->master_ipv4 || strnlen(p->master_ipv4, INET_ADDRSTRLEN) >= INET_ADDRSTRLEN ||
        inet_pton(AF_INET, p->master_ipv4, &ip) != 1 ||
        p->master_port < 1024 || p->master_port > 65535 || p->gid_index > 255 ||
        p->timeout_ms < 100 || p->timeout_ms > 120000) {
        copy_error(error_buffer, error_bytes, "invalid ABI/rank/arena/pointers/connection parameters");
        return CPU_RDMA_INVALID;
    }
    if (!fp_state_ok() || !reduction_avx2_available()) {
        copy_error(error_buffer, error_bytes, "requires AVX2, RNE and disabled FTZ/DAZ in calling thread");
        return CPU_RDMA_UNSUPPORTED;
    }
    cpu_rdma_ctx *s = calloc(1, sizeof(*s));
    if (!s) { copy_error(error_buffer, error_bytes, "allocate context failed"); return CPU_RDMA_SYSTEM; }
    s->rank = (int)p->rank; s->gid_index = (int)p->gid_index; s->port = (int)p->master_port;
    s->timeout_ms = p->timeout_ms; s->deadline = now_sec() + p->timeout_ms / 1000.0;
    s->memory = arena; s->listener = -1;
    for (int r = 0; r < NR; ++r) s->fd[r] = -1;
    memcpy(s->device, p->device, strlen(p->device) + 1);
    memcpy(s->master, p->master_ipv4, strlen(p->master_ipv4) + 1);
    memcpy(s->run_id, p->run_id, 33);
    /* Initialize receive slots only; preserve caller's input/output contents. */
    memset(arena, 0, CPU_RDMA_INPUT_OFFSET);
    int setup_result = setup(s);
    if (setup_result || expired(s)) {
        int code = expired(s) ? CPU_RDMA_TIMEOUT : CPU_RDMA_SYSTEM;
        snprintf(s->error, sizeof(s->error), "%s failed errno%d", s->stage ? s->stage : "setup", errno);
        copy_error(error_buffer, error_bytes, s->error);
        if (cpu_rdma_destroy(s)) {
            *out = s;
            copy_error(error_buffer, error_bytes, s->error);
        }
        return code;
    }
    s->ready = 1; *out = s;
    return CPU_RDMA_OK;
}
int cpu_rdma_run(cpu_rdma_ctx *s, const void *input_host, void *output_host,
                 size_t values, cpu_rdma_mode mode) {
    if (!s) return CPU_RDMA_INVALID;
    if (!s->ready || s->poisoned) return CPU_RDMA_POISONED;
    if (input_host != (char *)s->memory + CPU_RDMA_INPUT_OFFSET ||
        output_host != (char *)s->memory + CPU_RDMA_OUTPUT_OFFSET || values != CPU_RDMA_VALUES ||
        (mode != CPU_RDMA_FP32_THEN_BF16 && mode != CPU_RDMA_BF16_EACH_ADD &&
         mode != CPU_RDMA_RCCL_RING4_BF16) ||
        s->sequence >= UINT32_MAX - 1u) {
        s->poisoned = 1;
        copy_error(s->error, sizeof(s->error), "invalid fixed input/output alias, count, mode or exhausted sequence");
        return CPU_RDMA_INVALID;
    }
    if (!fp_state_ok()) {
        s->poisoned = 1;
        copy_error(s->error, sizeof(s->error), "calling thread FP control state requires RNE and FTZ/DAZ disabled");
        return CPU_RDMA_UNSUPPORTED;
    }
    s->deadline = now_sec() + s->timeout_ms / 1000.0;
    int run_result = collective(s, input_host, output_host, (int)mode);
    if (run_result || expired(s)) {
        s->poisoned = 1;
        int code = expired(s) ? CPU_RDMA_TIMEOUT : CPU_RDMA_SYSTEM;
        if (!*s->error) snprintf(s->error, sizeof(s->error), "collective failed rank%d seq%u errno%d", s->rank, s->sequence, errno);
        return code;
    }
    return CPU_RDMA_OK;
}
int cpu_rdma_numerics_selftest(char *error_buffer, size_t error_bytes) {
    copy_error(error_buffer, error_bytes, "");
    if (!fp_state_ok() || !reduction_avx2_available()) {
        copy_error(error_buffer, error_bytes, "numeric selftest requires AVX2, RNE and disabled FTZ/DAZ");
        return CPU_RDMA_UNSUPPORTED;
    }
    uint16_t data[4][64], output[64];
    const uint16_t *a[4] = {data[0], data[1], data[2], data[3]};
    for (unsigned pattern = 0; pattern < PERIOD + 8; ++pattern) {
        unsigned count = pattern < PERIOD ? N : 65536u;
        for (unsigned base = 0; base < count; base += 64) {
            unsigned chunk = count - base < 64 ? count - base : 64;
            for (unsigned j = 0; j < chunk; ++j) for (unsigned r = 0; r < 4; ++r) {
                unsigned i = base + j, edge = pattern - PERIOD;
                uint16_t v;
                if (pattern < PERIOD) v = input_bits(r, pattern, i);
                else if (edge < 4) v = r == edge ? (uint16_t)i : (r & 1u) ? 0xbf80 : 0x3f80;
                else if (edge == 4) v = (uint16_t)(i + r * 16381u);
                else if (edge == 5) v = (uint16_t)i;
                else if (edge == 6) v = (r == 0 || r == 2) ? (uint16_t)(i ^ (r == 2 ? 0x8000u : 0)) : 0x3f80;
                else v = (uint16_t)((i * 40503u + r * 49157u) ^ ((i + r) >> 3));
                data[r][j] = v;
            }
            for (int mode = 0; mode < 2; ++mode) {
                reduce_avx2(a, output, chunk, mode);
                for (unsigned j = 0; j < chunk; ++j) {
                    uint16_t v[4] = {data[0][j], data[1][j], data[2][j], data[3][j]};
                    if (output[j] != reduce_four(v, mode)) {
                        snprintf(error_buffer, error_buffer && error_bytes ? error_bytes : 0,
                            "SIMD/scalar mismatch pattern%u index%u mode%d", pattern, base + j, mode);
                        return CPU_RDMA_SYSTEM;
                    }
                }
            }
        }
    }
    if (to_bf16(1.00390625f) != 0x3f80 || to_bf16(1.01171875f) != 0x3f82 ||
        to_bf16(-1.00390625f) != 0xbf80 || to_bf16(-1.01171875f) != 0xbf82) {
        copy_error(error_buffer, error_bytes, "BF16 ties-to-even golden failed"); return CPU_RDMA_SYSTEM;
    }
    return CPU_RDMA_OK;
}
