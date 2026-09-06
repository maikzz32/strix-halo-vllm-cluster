// Host-only HIP API harness. The graph callback never calls HIP or Python.
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t count = 10240; // BF16, exactly 20 KiB in each transfer direction.
constexpr size_t bytes = count * sizeof(uint16_t);
struct Context {
    uint16_t* staging = nullptr;
    bool copies = false;
    unsigned input_phase = 0; // Changed only after stream completion, outside the graph.
    std::atomic<uint64_t> calls{0}, bad_inputs{0}, marker{0};
};
uint16_t input_bits(size_t i, unsigned phase) { return uint16_t(0x3f80u + ((i + phase) % 128)); }
float decode(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
void callback(void* opaque) {
    auto* c = static_cast<Context*>(opaque);
    // No allocation, HIP call, Python/GIL, I/O, network, or blocking wait here.
    if (c->copies) {
        uint64_t bad = 0;
        for (size_t i = 0; i < count; ++i) {
            bad += c->staging[i] != input_bits(i, c->input_phase);
            c->staging[i] ^= 0x8000u; // Exact BF16 negation of finite positive input.
        }
        c->bad_inputs.fetch_add(bad, std::memory_order_relaxed);
    }
    const auto done = c->calls.fetch_add(1, std::memory_order_relaxed) + 1;
    c->marker.store(done + 42, std::memory_order_release);
}
void hip_check(hipError_t error, const char* api) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(api) + ": " + hipGetErrorName(error) +
                                 " (" + std::to_string(int(error)) + ") " + hipGetErrorString(error));
}
#define HIP(call) hip_check((call), #call)
struct Resources {
    Context context;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    uint16_t *input = nullptr, *output = nullptr;
    bool capturing = false;
    ~Resources() {
        if (capturing) {
            hipGraph_t abandoned = nullptr;
            hipStreamEndCapture(stream, &abandoned);
            if (abandoned) hipGraphDestroy(abandoned);
        }
        if (stream) hipStreamSynchronize(stream);
        if (executable) hipGraphExecDestroy(executable);
        if (graph) hipGraphDestroy(graph);
        if (begin) hipEventDestroy(begin);
        if (end) hipEventDestroy(end);
        if (input) hipFree(input);
        if (output) hipFree(output);
        if (context.staging) hipHostFree(context.staging);
        if (stream) hipStreamDestroy(stream);
    }
};
}

extern "C" {
struct HostGraphResult {
    int mode, replay_count, sample_count;
    int node_count, host_node_count, memcpy_node_count;
    uint64_t callback_count, expected_callback_count, bad_input_count, resident_bytes;
    double wall_us[5], hip_event_us[5], max_abs_error;
    int finite, exact, capture_supported, runtime_version;
    char error[512];
};

int run_host_graph_case(int mode, int replays, int samples, HostGraphResult* out) {
    if (!out) return 2;
    std::memset(out, 0, sizeof(*out));
    out->mode = mode;
    out->replay_count = replays;
    out->sample_count = samples;
    try {
        if (mode < 0 || mode > 2 || replays < 5 || replays > 100 || samples < 1 || samples > 5)
            throw std::runtime_error("Invalid bounded arguments");
        Resources r;
        r.context.copies = mode != 0;
        HIP(hipRuntimeGetVersion(&out->runtime_version));
        HIP(hipSetDevice(0));
        HIP(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking));
        HIP(hipEventCreate(&r.begin));
        HIP(hipEventCreate(&r.end));
        if (r.context.copies) {
            HIP(hipHostMalloc(reinterpret_cast<void**>(&r.context.staging), bytes, hipHostMallocDefault));
            HIP(hipMalloc(reinterpret_cast<void**>(&r.input), bytes));
            HIP(hipMalloc(reinterpret_cast<void**>(&r.output), bytes));
            for (size_t i = 0; i < count; ++i) r.context.staging[i] = input_bits(i, 0);
            HIP(hipMemcpy(r.input, r.context.staging, bytes, hipMemcpyHostToDevice));
            HIP(hipMemset(r.output, 0, bytes));
            std::fill_n(r.context.staging, count, uint16_t(0));
            out->resident_bytes = 3 * bytes;
        }
        HIP(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal));
        r.capturing = true;
        if (r.context.copies)
            HIP(hipMemcpyAsync(r.context.staging, r.input, bytes, hipMemcpyDeviceToHost, r.stream));
        if (mode != 2) HIP(hipLaunchHostFunc(r.stream, callback, &r.context));
        if (r.context.copies)
            HIP(hipMemcpyAsync(r.output, r.context.staging, bytes, hipMemcpyHostToDevice, r.stream));
        hipError_t capture_error = hipStreamEndCapture(r.stream, &r.graph);
        r.capturing = false;
        hip_check(capture_error, "hipStreamEndCapture");
        size_t nodes = 0;
        HIP(hipGraphGetNodes(r.graph, nullptr, &nodes));
        std::vector<hipGraphNode_t> node_list(nodes);
        HIP(hipGraphGetNodes(r.graph, node_list.data(), &nodes));
        out->node_count = int(nodes);
        for (auto node : node_list) {
            hipGraphNodeType type;
            HIP(hipGraphNodeGetType(node, &type));
            out->host_node_count += type == hipGraphNodeTypeHost;
            out->memcpy_node_count += type == hipGraphNodeTypeMemcpy;
        }
        if (out->host_node_count != (mode == 2 ? 0 : 1) || out->memcpy_node_count != (mode ? 2 : 0))
            throw std::runtime_error("Captured graph lacks expected host/memcpy nodes");
        if (r.context.calls.load() != 0)
            throw std::runtime_error("Callback executed during capture instead of replay");
        HIP(hipGraphInstantiate(&r.executable, r.graph, nullptr, nullptr, 0));
        out->capture_supported = 1;
        std::vector<uint16_t> readback(mode ? count : 0);
        uint64_t expected = 0;
        auto change_input = [&](unsigned phase) {
            HIP(hipStreamSynchronize(r.stream));
            r.context.input_phase = phase;
            if (mode) {
                for (size_t i = 0; i < count; ++i)
                    r.context.staging[i] = input_bits(i, phase);
                HIP(hipMemcpy(r.input, r.context.staging, bytes, hipMemcpyHostToDevice));
                std::fill_n(r.context.staging, count, uint16_t(0));
                HIP(hipMemset(r.output, 0, bytes));
            }
        };
        auto verify = [&]() {
            HIP(hipStreamSynchronize(r.stream));
            if (r.context.calls.load() != expected ||
                r.context.marker.load() != (mode == 2 ? 0 : expected + 42) ||
                r.context.bad_inputs.load() != 0)
                throw std::runtime_error("CPU callback count/marker/input correctness failure");
            if (mode) {
                HIP(hipMemcpy(readback.data(), r.output, bytes, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; ++i) {
                    const float actual = decode(readback[i]);
                    const uint16_t bits = input_bits(i, r.context.input_phase);
                    const uint16_t expected_bits = bits ^ (mode == 1 ? 0x8000u : 0u);
                    const float reference = mode == 1 ? -decode(bits) : decode(bits);
                    out->max_abs_error = std::max(out->max_abs_error, double(std::abs(actual - reference)));
                    if (!std::isfinite(actual) || readback[i] != expected_bits)
                        throw std::runtime_error("BF16 roundtrip does not match exact FP32 identity/negation");
                }
            }
        };
        // Validate each replay on changed nonzero data before any timing.
        for (int i = 0; i < 4; ++i) {
            change_input(unsigned(i * 13));
            HIP(hipGraphLaunch(r.executable, r.stream));
            expected += mode != 2;
            verify();
        }
        for (int sample = 0; sample < samples; ++sample) {
            change_input(unsigned(7 + sample * 17));
            const auto wall_start = std::chrono::steady_clock::now();
            HIP(hipEventRecord(r.begin, r.stream));
            for (int i = 0; i < replays; ++i) {
                HIP(hipGraphLaunch(r.executable, r.stream));
                expected += mode != 2;
            }
            HIP(hipEventRecord(r.end, r.stream));
            HIP(hipEventSynchronize(r.end));
            const auto wall_end = std::chrono::steady_clock::now();
            float ms = 0;
            HIP(hipEventElapsedTime(&ms, r.begin, r.end));
            out->wall_us[sample] = std::chrono::duration<double, std::micro>(wall_end - wall_start).count() / replays;
            out->hip_event_us[sample] = double(ms) * 1000 / replays;
            verify(); // Outside timed region, including complete 10240-element readback.
        }
        out->callback_count = r.context.calls.load();
        out->expected_callback_count = expected;
        out->bad_input_count = r.context.bad_inputs.load();
        out->finite = out->exact = 1;
        return 0;
    } catch (const std::exception& e) {
        std::snprintf(out->error, sizeof(out->error), "%s", e.what());
        return 1;
    }
}
}
