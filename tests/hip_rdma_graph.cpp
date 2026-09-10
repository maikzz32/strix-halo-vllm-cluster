// Isolated fixed-shape four-rank HIP graph + CPU verbs bridge. No serving hooks.
#include "cpu_rdma_transport.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t N = CPU_RDMA_VALUES, BYTES = CPU_RDMA_BYTES;
using Clock = std::chrono::steady_clock;
uint16_t input_bits(unsigned rank, unsigned phase, unsigned i) {
    static const uint16_t cancellation[4] = {0x4380, 0x3f80, 0xc380, 0x3f80};
    static const uint16_t small[4] = {0x3f80, 0x3b80, 0xbf80, 0x3b80};
    if (i % 16 == 0) return cancellation[rank];
    if (i % 16 == 1) return small[rank];
    uint32_t h = i * 0x9e3779b1u ^ ((phase % 32) + 1) * 0x85ebca6bu ^ (rank + 1) * 0xc2b2ae35u;
    h ^= h >> 16;
    return uint16_t(((h >> 15) & 0x8000u) | ((123 + ((h >> 24) % 9)) << 7) | (h & 127));
}
float from_bf16(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, 4);
    return value;
}
uint16_t to_bf16(float value) {
    uint32_t word;
    std::memcpy(&word, &value, 4);
    if ((word & 0x7f800000u) == 0x7f800000u)
        return (word & 0x7fffffu) ? uint16_t(((word >> 16) & 0x8000u) | 0x7fc0u) : uint16_t(word >> 16);
    return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
uint16_t reference(unsigned phase, unsigned i, int mode) {
    float sum = from_bf16(input_bits(0, phase, i));
    for (unsigned rank = 1; rank < 4; ++rank) {
        sum = sum + from_bf16(input_bits(rank, phase, i));
        if (mode) sum = from_bf16(to_bf16(sum));
    }
    return to_bf16(sum);
}
void hip_check(hipError_t error, const char* api) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(api) + ": " + hipGetErrorName(error) +
                                 " (" + std::to_string(int(error)) + ") " + hipGetErrorString(error));
}
#define HIP(call) hip_check((call), #call)
struct Callback {
    cpu_rdma_ctx* transport = nullptr;
    uint16_t *input = nullptr, *output = nullptr;
    cpu_rdma_mode mode = CPU_RDMA_FP32_THEN_BF16;
    Clock::time_point deadline;
    uint32_t main_control = 0;
    std::atomic<uint64_t> calls{0}, successes{0};
    std::atomic<int> sticky{0}, inject_bad_count{0};
    int selftest_rc = -999, rounding_before = -1, rounding_after = -1;
    uint32_t mxcsr_before = 0, mxcsr_after = 0;
    char error[512] = {};
};
void latch(Callback* cb, int code, const char* text) {
    std::snprintf(cb->error, sizeof(cb->error), "%s", text ? text : "Unknown native error");
    cb->sticky.store(code ? code : CPU_RDMA_SYSTEM, std::memory_order_release);
}
void native_callback(void* user) {
    auto* cb = static_cast<Callback*>(user);
    const uint64_t call = cb->calls.fetch_add(1, std::memory_order_relaxed);
    // No HIP/Python, allocations, TCP barriers, logs, or global signal changes.
    if (call == 0) {
        cb->mxcsr_before = _mm_getcsr();
        cb->rounding_before = std::fegetround();
        char error[512] = {};
        cb->selftest_rc = cpu_rdma_numerics_selftest(error, sizeof(error));
        cb->mxcsr_after = _mm_getcsr();
        cb->rounding_after = std::fegetround();
        if (cb->selftest_rc != CPU_RDMA_OK) latch(cb, cb->selftest_rc, error);
        else if ((cb->mxcsr_before & ~63u) != cb->main_control ||
                 (cb->mxcsr_after & ~63u) != cb->main_control ||
                 cb->rounding_before != FE_TONEAREST || cb->rounding_after != FE_TONEAREST)
            latch(cb, CPU_RDMA_UNSUPPORTED, "Main/callback FP control state differs");
    }
    if (!cb->sticky.load(std::memory_order_acquire)) {
        if (Clock::now() >= cb->deadline) latch(cb, CPU_RDMA_TIMEOUT, "Overall graph deadline exceeded");
        else {
            const size_t values = cb->inject_bad_count.load() ? N - 1 : N;
            const int rc = cpu_rdma_run(cb->transport, cb->input, cb->output, values, cb->mode);
            if (rc != CPU_RDMA_OK) latch(cb, rc, cpu_rdma_last_error(cb->transport));
            else cb->successes.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (cb->sticky.load(std::memory_order_acquire))
        std::fill_n(cb->output, N, uint16_t(0x7fc0)); // H2D/validation cannot pass on stale data.
}
}

extern "C" {
struct HipRdmaOptions {
    int rank, gid_index, port, deadline_seconds, mode, iterations, samples, warmup;
    const char *device, *master, *run_id;
};
struct HipRdmaResult {
    int rank, mode, iterations, samples, warmup, runtime_version;
    int node_count, host_nodes, memcpy_nodes, main_selftest_rc, callback_selftest_rc;
    int main_rounding_before, main_rounding_after, callback_rounding_before, callback_rounding_after;
    uint32_t main_mxcsr_before, main_mxcsr_after, callback_mxcsr_before, callback_mxcsr_after;
    uint64_t callback_calls, successful_collectives, expected_collectives, arena_bytes;
    int direct_pinned_checks, graph_numeric_checks, exact, finite, pinned_mr_ok;
    int negative_test_passed, expected_negative_code, cleanup_rc, arena_released;
    double setup_seconds, total_seconds, max_abs_error, wall_us[5], hip_event_us[5];
    char error[512], expected_negative_error[512];
};
}

namespace {
struct Resources {
    Callback* cb = new Callback;
    void* arena = nullptr;
    uint16_t *device_input = nullptr, *device_output = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    bool capturing = false;
    // Failure intentionally retains borrowed memory/callback context until process exit.
    int close(HipRdmaResult* out) {
        if (capturing) {
            hipGraph_t abandoned = nullptr;
            const auto rc = hipStreamEndCapture(stream, &abandoned);
            capturing = false;
            if (abandoned && hipGraphDestroy(abandoned) != hipSuccess) return -101;
            if (rc != hipSuccess && rc != hipErrorStreamCaptureInvalidated) return -102;
        }
        if (stream && hipStreamSynchronize(stream) != hipSuccess) return -103;
        if (executable && hipGraphExecDestroy(executable) != hipSuccess) return -104;
        executable = nullptr;
        if (graph && hipGraphDestroy(graph) != hipSuccess) return -105;
        graph = nullptr;
        if (cb->transport) {
            const int rc = cpu_rdma_destroy(cb->transport);
            if (rc != CPU_RDMA_OK) return rc; // MR may still own arena: do not free it.
            cb->transport = nullptr;
        }
        if (begin && hipEventDestroy(begin) != hipSuccess) return -106;
        begin = nullptr;
        if (end && hipEventDestroy(end) != hipSuccess) return -107;
        end = nullptr;
        if (device_input && hipFree(device_input) != hipSuccess) return -108;
        device_input = nullptr;
        if (device_output && hipFree(device_output) != hipSuccess) return -109;
        device_output = nullptr;
        if (arena && hipHostFree(arena) != hipSuccess) return -110;
        arena = nullptr;
        out->arena_released = 1;
        if (stream && hipStreamDestroy(stream) != hipSuccess) return -111;
        stream = nullptr;
        delete cb;
        cb = nullptr;
        return 0;
    }
};
}

extern "C" int run_hip_rdma_graph_case(const HipRdmaOptions* o, HipRdmaResult* out) {
    if (!o || !out) return 2;
    std::memset(out, 0, sizeof(*out));
    out->rank = o->rank; out->mode = o->mode; out->iterations = o->iterations;
    out->samples = o->samples; out->warmup = o->warmup;
    const auto started = Clock::now();
    Resources r;
    int status = 0;
    try {
        if (o->rank < 0 || o->rank > 3 || o->mode < 0 || o->mode > 1 ||
            o->deadline_seconds < 10 || o->deadline_seconds > 120 ||
            o->iterations < 1 || o->iterations > 128 || o->samples < 1 || o->samples > 5 ||
            o->warmup < 1 || o->warmup > 16)
            throw std::runtime_error("Invalid bounded test options");
        r.cb->deadline = started + std::chrono::seconds(o->deadline_seconds);
        r.cb->mode = static_cast<cpu_rdma_mode>(o->mode);
        HIP(hipRuntimeGetVersion(&out->runtime_version));
        HIP(hipSetDevice(0));
        HIP(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking));
        HIP(hipEventCreate(&r.begin));
        HIP(hipEventCreate(&r.end));
        HIP(hipHostMalloc(&r.arena, CPU_RDMA_ARENA_BYTES, hipHostMallocDefault));
        out->arena_bytes = CPU_RDMA_ARENA_BYTES;
        if ((reinterpret_cast<uintptr_t>(r.arena) & 63u) != 0)
            throw std::runtime_error("HIP host allocation does not satisfy 64-byte alignment");
        r.cb->input = reinterpret_cast<uint16_t*>(static_cast<char*>(r.arena) + CPU_RDMA_INPUT_OFFSET);
        r.cb->output = reinterpret_cast<uint16_t*>(static_cast<char*>(r.arena) + CPU_RDMA_OUTPUT_OFFSET);
        HIP(hipMalloc(reinterpret_cast<void**>(&r.device_input), BYTES));
        HIP(hipMalloc(reinterpret_cast<void**>(&r.device_output), BYTES));
        if (r.device_input == r.device_output) throw std::runtime_error("Device buffers unexpectedly alias");
        out->main_mxcsr_before = _mm_getcsr();
        out->main_rounding_before = std::fegetround();
        char native_error[512] = {};
        out->main_selftest_rc = cpu_rdma_numerics_selftest(native_error, sizeof(native_error));
        out->main_mxcsr_after = _mm_getcsr();
        out->main_rounding_after = std::fegetround();
        if (out->main_selftest_rc != CPU_RDMA_OK) throw std::runtime_error(native_error);
        r.cb->main_control = out->main_mxcsr_after & ~63u;
        if ((out->main_mxcsr_before & ~63u) != r.cb->main_control)
            throw std::runtime_error("Numerical selftest modified main FP control state");
        cpu_rdma_params params = {CPU_RDMA_ABI_VERSION, uint32_t(o->rank), CPU_RDMA_WORLD_SIZE,
            o->run_id, o->device, o->master, uint32_t(o->port), uint32_t(o->gid_index),
            uint32_t(std::min(o->deadline_seconds, 5) * 1000)};
        const int create_rc = cpu_rdma_create(&params, r.arena, CPU_RDMA_ARENA_BYTES,
                                              &r.cb->transport, native_error, sizeof(native_error));
        if (create_rc != CPU_RDMA_OK) throw std::runtime_error(std::string("cpu_rdma_create: ") + native_error);
        out->pinned_mr_ok = 1;
        std::vector<uint16_t> readback(N);
        auto set_input = [&](unsigned phase) {
            HIP(hipStreamSynchronize(r.stream));
            if (Clock::now() >= r.cb->deadline) throw std::runtime_error("Overall deadline exceeded");
            for (unsigned i = 0; i < N; ++i) r.cb->input[i] = input_bits(unsigned(o->rank), phase, i);
            HIP(hipMemcpy(r.device_input, r.cb->input, BYTES, hipMemcpyHostToDevice));
            std::fill_n(r.cb->input, N, uint16_t(0));
            std::fill_n(r.cb->output, N, uint16_t(0x7fc0));
            HIP(hipMemset(r.device_output, 0, BYTES));
        };
        auto verify = [&](unsigned phase) {
            HIP(hipStreamSynchronize(r.stream));
            if (r.cb->sticky.load()) throw std::runtime_error(r.cb->error);
            HIP(hipMemcpy(readback.data(), r.device_output, BYTES, hipMemcpyDeviceToHost));
            for (unsigned i = 0; i < N; ++i) {
                const uint16_t expected = reference(phase, i, o->mode);
                const float value = from_bf16(readback[i]);
                out->max_abs_error = std::max(out->max_abs_error, double(std::abs(value - from_bf16(expected))));
                if (!std::isfinite(value) || readback[i] != expected)
                    throw std::runtime_error("Device BF16 output/reference mismatch at element " + std::to_string(i));
            }
        };
        // Prove exact HIP-pinned MR path before capture, with four changed sources.
        for (unsigned phase = 0; phase < 4; ++phase) {
            set_input(phase);
            HIP(hipMemcpyAsync(r.cb->input, r.device_input, BYTES, hipMemcpyDeviceToHost, r.stream));
            HIP(hipStreamSynchronize(r.stream));
            const int rc = cpu_rdma_run(r.cb->transport, r.cb->input, r.cb->output, N, r.cb->mode);
            if (rc != CPU_RDMA_OK) throw std::runtime_error(cpu_rdma_last_error(r.cb->transport));
            HIP(hipMemcpyAsync(r.device_output, r.cb->output, BYTES, hipMemcpyHostToDevice, r.stream));
            verify(phase);
            ++out->direct_pinned_checks;
        }
        HIP(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal));
        r.capturing = true;
        HIP(hipMemcpyAsync(r.cb->input, r.device_input, BYTES, hipMemcpyDeviceToHost, r.stream));
        HIP(hipLaunchHostFunc(r.stream, native_callback, r.cb));
        HIP(hipMemcpyAsync(r.device_output, r.cb->output, BYTES, hipMemcpyHostToDevice, r.stream));
        const auto capture_rc = hipStreamEndCapture(r.stream, &r.graph);
        r.capturing = false;
        hip_check(capture_rc, "hipStreamEndCapture");
        size_t count = 0;
        HIP(hipGraphGetNodes(r.graph, nullptr, &count));
        std::vector<hipGraphNode_t> nodes(count);
        HIP(hipGraphGetNodes(r.graph, nodes.data(), &count));
        out->node_count = int(count);
        for (auto node : nodes) {
            hipGraphNodeType type;
            HIP(hipGraphNodeGetType(node, &type));
            out->host_nodes += type == hipGraphNodeTypeHost;
            out->memcpy_nodes += type == hipGraphNodeTypeMemcpy;
        }
        if (count != 3 || out->host_nodes != 1 || out->memcpy_nodes != 2 || r.cb->calls.load())
            throw std::runtime_error("Graph node types/capture side effects unexpected");
        HIP(hipGraphInstantiate(&r.executable, r.graph, nullptr, nullptr, 0));
        uint64_t expected = 0;
        for (int i = 0; i < o->warmup; ++i) {
            const unsigned phase = unsigned(4 + i);
            set_input(phase);
            HIP(hipGraphLaunch(r.executable, r.stream));
            ++expected;
            verify(phase);
            ++out->graph_numeric_checks;
        }
        out->setup_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (int sample = 0; sample < o->samples; ++sample) {
            const unsigned phase = unsigned(17 + 3 * sample);
            set_input(phase);
            const auto begin = Clock::now();
            HIP(hipEventRecord(r.begin, r.stream));
            for (int replay = 0; replay < o->iterations; ++replay) {
                HIP(hipGraphLaunch(r.executable, r.stream));
                ++expected;
            }
            HIP(hipEventRecord(r.end, r.stream));
            HIP(hipEventSynchronize(r.end));
            const auto end = Clock::now();
            float ms = 0;
            HIP(hipEventElapsedTime(&ms, r.begin, r.end));
            out->wall_us[sample] = std::chrono::duration<double, std::micro>(end - begin).count() / o->iterations;
            out->hip_event_us[sample] = double(ms) * 1000 / o->iterations;
            verify(phase);
            ++out->graph_numeric_checks;
            if (r.cb->calls.load() != expected || r.cb->successes.load() != expected)
                throw std::runtime_error("Graph callback/collective count mismatch");
        }
        out->expected_collectives = expected;
        out->successful_collectives = r.cb->successes.load();
        out->exact = out->finite = 1;
        // Coordinated terminal negative case: invalid count must fail before any send.
        // Next replay must retain the sticky error, never output stale valid data.
        r.cb->inject_bad_count.store(1);
        for (int i = 0; i < 2; ++i) {
            HIP(hipGraphLaunch(r.executable, r.stream));
            HIP(hipStreamSynchronize(r.stream));
            r.cb->inject_bad_count.store(0);
            HIP(hipMemcpy(readback.data(), r.device_output, BYTES, hipMemcpyDeviceToHost));
            if (!r.cb->sticky.load() || r.cb->successes.load() != expected ||
                r.cb->calls.load() != expected + unsigned(i + 1) ||
                !std::all_of(readback.begin(), readback.end(), [](uint16_t v) { return v == 0x7fc0; }))
                throw std::runtime_error("Negative/sticky-error test did not fail closed");
        }
        out->negative_test_passed = 1;
        out->expected_negative_code = r.cb->sticky.load();
        std::snprintf(out->expected_negative_error, sizeof(out->expected_negative_error), "%s", r.cb->error);
    } catch (const std::exception& error) {
        status = 1;
        std::snprintf(out->error, sizeof(out->error), "%s", error.what());
    }
    out->callback_calls = r.cb->calls.load();
    out->callback_selftest_rc = r.cb->selftest_rc;
    out->callback_mxcsr_before = r.cb->mxcsr_before;
    out->callback_mxcsr_after = r.cb->mxcsr_after;
    out->callback_rounding_before = r.cb->rounding_before;
    out->callback_rounding_after = r.cb->rounding_after;
    out->cleanup_rc = r.close(out);
    if (out->cleanup_rc != 0) {
        status = 1;
        if (!out->error[0]) std::snprintf(out->error, sizeof(out->error), "Cleanup failed (%d); borrowed buffers retained", out->cleanup_rc);
    }
    out->total_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return status;
}

extern "C" size_t hip_rdma_options_size() { return sizeof(HipRdmaOptions); }
extern "C" size_t hip_rdma_result_size() { return sizeof(HipRdmaResult); }
