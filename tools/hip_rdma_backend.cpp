// Experimental fixed-shape TP4 backend. No installation or runtime hook.
// Build cpu_rdma_transport.c as C, then link this C++ file with HIP/ibverbs.
// Caller must serialize collective order and graph replays across all ranks.
#include "cpu_rdma_transport.h"
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <unistd.h>

namespace {
constexpr size_t BYTES = CPU_RDMA_BYTES;
struct Backend;
struct Descriptor {
    Backend* owner = nullptr;
    void *input = nullptr, *output = nullptr;
    hipEvent_t completed = nullptr;
    bool used = false, captured = false;
};
struct Backend {
    cpu_rdma_ctx* transport = nullptr;
    void *arena = nullptr, *staging = nullptr;
    Descriptor* descriptors = nullptr;
    unsigned count = 0, next_eager = 0, next_capture = 0;
    int device = -1, rank = -1;
    cpu_rdma_mode mode = CPU_RDMA_FP32_THEN_BF16;
    bool has_capture = false, ready = false, tested_callback = false;
    std::atomic<bool> active{false};
    std::mutex enqueue_mutex;
};
int error(char* buffer, size_t size, const char* message, int code = -1) {
    if (buffer && size) std::snprintf(buffer, size, "%s", message);
    return code;
}
int hip_error(char* buffer, size_t size, hipError_t rc, const char* api) {
    if (buffer && size)
        std::snprintf(buffer, size, "%s failed: %s (%d)", api, hipGetErrorName(rc), int(rc));
    return -1000 - int(rc);
}
[[noreturn]] void fail_stop(Backend* backend, const char* message) noexcept {
    char line[768];
    const int n = std::snprintf(line, sizeof(line),
        "HIP_RDMA_BACKEND_FATAL rank=%d: %s\n", backend ? backend->rank : -1, message);
    if (n > 0) {
        const size_t size = size_t(n) < sizeof(line) ? size_t(n) : sizeof(line) - 1;
        // Best-effort bounded-size stderr write; no Python/GIL, HIP or allocation.
        (void)!write(STDERR_FILENO, line, size);
    }
    _exit(86);  // Never return to GPU consumers with an invalid output.
}
void callback(void* user) noexcept {
    auto* descriptor = static_cast<Descriptor*>(user);
    Backend* backend = descriptor->owner;
    bool expected = false;
    if (!backend->active.compare_exchange_strong(expected, true, std::memory_order_acquire))
        fail_stop(backend, "concurrent callbacks violate serialized TP collective contract");
    if (!backend->tested_callback) {
        char message[512] = {};
        if (cpu_rdma_numerics_selftest(message, sizeof(message)) != CPU_RDMA_OK)
            fail_stop(backend, message);
        backend->tested_callback = true;
    }
    auto* input = static_cast<char*>(backend->arena) + CPU_RDMA_INPUT_OFFSET;
    auto* output = static_cast<char*>(backend->arena) + CPU_RDMA_OUTPUT_OFFSET;
    std::memcpy(input, descriptor->input, BYTES);
    const int rc = cpu_rdma_run(backend->transport, input, output, CPU_RDMA_VALUES, backend->mode);
    if (rc != CPU_RDMA_OK) fail_stop(backend, cpu_rdma_last_error(backend->transport));
    std::memcpy(descriptor->output, output, BYTES);
    backend->active.store(false, std::memory_order_release);
}
}

extern "C" unsigned hip_rdma_backend_abi_version() { return 1; }
extern "C" size_t hip_rdma_backend_params_size() { return sizeof(cpu_rdma_params); }

// On any failure a non-null *out retains all partial resources. Caller must
// close it outside capture or terminate its own experimental worker process.
extern "C" int hip_rdma_backend_create(const cpu_rdma_params* params, int device,
    int mode, unsigned slots, void** out, char* message, size_t message_bytes) {
    if (out) *out = nullptr;
    if (!params || !out || device < 0 || (mode < 0 || mode > 2) || slots < 128 || slots > 8192)
        return error(message, message_bytes, "invalid backend options");
    Backend* backend = new (std::nothrow) Backend;
    if (!backend) return error(message, message_bytes, "allocate backend failed");
    *out = backend;
    backend->device = device;
    backend->rank = int(params->rank);
    backend->mode = static_cast<cpu_rdma_mode>(mode);
    backend->count = slots;
#define CREATE_HIP(call) do { const auto rc = (call); if (rc != hipSuccess) \
    return hip_error(message, message_bytes, rc, #call); } while (0)
    CREATE_HIP(hipSetDevice(device));
    hipStreamCaptureStatus capture;
    CREATE_HIP(hipStreamIsCapturing(nullptr, &capture));
    if (capture != hipStreamCaptureStatusNone)
        return error(message, message_bytes, "backend create must precede graph capture");
    CREATE_HIP(hipHostMalloc(&backend->arena, CPU_RDMA_ARENA_BYTES, hipHostMallocDefault));
    CREATE_HIP(hipHostMalloc(&backend->staging, size_t(slots) * 2 * BYTES, hipHostMallocDefault));
    backend->descriptors = new (std::nothrow) Descriptor[slots];
    if (!backend->descriptors) return error(message, message_bytes, "allocate descriptors failed");
    for (unsigned i = 0; i < slots; ++i) {
        Descriptor& descriptor = backend->descriptors[i];
        descriptor.owner = backend;
        descriptor.input = static_cast<char*>(backend->staging) + size_t(i) * 2 * BYTES;
        descriptor.output = static_cast<char*>(descriptor.input) + BYTES;
        CREATE_HIP(hipEventCreateWithFlags(&descriptor.completed, hipEventDisableTiming));
    }
    if (cpu_rdma_numerics_selftest(message, message_bytes) != CPU_RDMA_OK)
        return error(message, message_bytes, "main-thread numerical selftest failed");
    const int rc = cpu_rdma_create(params, backend->arena, CPU_RDMA_ARENA_BYTES,
                                   &backend->transport, message, message_bytes);
    if (rc != CPU_RDMA_OK) return rc;
    backend->ready = true;
    return 0;
#undef CREATE_HIP
}

// All three operations use the explicitly supplied caller stream. Events are
// used only to reclaim eager descriptors; captured descriptors are never reused.
// No host/device allocation or TCP initialization occurs during enqueue/capture.
extern "C" int hip_rdma_backend_enqueue(void* handle, const void* input, void* output,
    size_t count, void* stream_pointer, char* message, size_t message_bytes) {
    auto* backend = static_cast<Backend*>(handle);
    if (!backend || !backend->ready || !input || !output || count != CPU_RDMA_VALUES)
        return error(message, message_bytes, "invalid fixed-shape backend enqueue");
    const auto a = reinterpret_cast<uintptr_t>(input), b = reinterpret_cast<uintptr_t>(output);
    if ((a >= b ? a - b : b - a) < BYTES)
        return error(message, message_bytes, "all_reduce must be out of place without overlapping storage");
    std::unique_lock<std::mutex> lock(backend->enqueue_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return error(message, message_bytes, "concurrent enqueue is unsupported");
    int current_device = -1;
    auto rc = hipGetDevice(&current_device);
    if (rc != hipSuccess) return hip_error(message, message_bytes, rc, "hipGetDevice");
    if (current_device != backend->device)
        return error(message, message_bytes, "current HIP device differs from backend device");
    const auto stream = reinterpret_cast<hipStream_t>(stream_pointer);
    hipStreamCaptureStatus capture;
    rc = hipStreamIsCapturing(stream, &capture);
    if (rc != hipSuccess) return hip_error(message, message_bytes, rc, "hipStreamIsCapturing");
    if (capture == hipStreamCaptureStatusInvalidated)
        return error(message, message_bytes, "caller capture is invalidated");
    const bool capturing = capture != hipStreamCaptureStatusNone;
    Descriptor* selected = nullptr;
    // Reserve three quarters for immutable captured descriptors. Eager warmup
    // must not consume every unused slot before the first graph is captured.
    const unsigned eager_count = backend->count / 4;
    const unsigned begin = capturing ? eager_count : 0;
    const unsigned length = capturing ? backend->count - eager_count : eager_count;
    unsigned& cursor = capturing ? backend->next_capture : backend->next_eager;
    for (unsigned offset = 0; offset < length; ++offset) {
        const unsigned index = begin + (cursor + offset) % length;
        Descriptor& descriptor = backend->descriptors[index];
        if (descriptor.captured) continue;
        if (descriptor.used) {
            // Capture only consumes unused slots; querying unrelated outstanding
            // eager events during capture would complicate capture semantics.
            if (capturing) continue;
            rc = hipEventQuery(descriptor.completed);
            if (rc == hipErrorNotReady) continue;
            if (rc != hipSuccess) return hip_error(message, message_bytes, rc, "hipEventQuery");
        }
        selected = &descriptor;
        cursor = (index - begin + 1) % length;
        break;
    }
    if (!selected) return error(message, message_bytes, "preallocated descriptor pool exhausted");
    selected->used = true;
    selected->captured = capturing;
    backend->has_capture |= capturing;
#define ENQUEUE_HIP(call) do { const auto result = (call); if (result != hipSuccess) { \
    hip_error(message, message_bytes, result, #call); \
    fail_stop(backend, message ? message : "partial enqueue failed"); } } while (0)
    ENQUEUE_HIP(hipMemcpyAsync(selected->input, input, BYTES, hipMemcpyDeviceToHost, stream));
    ENQUEUE_HIP(hipLaunchHostFunc(stream, callback, selected));
    ENQUEUE_HIP(hipMemcpyAsync(output, selected->output, BYTES, hipMemcpyHostToDevice, stream));
    if (!capturing) ENQUEUE_HIP(hipEventRecord(selected->completed, stream));
    return 0;
#undef ENQUEUE_HIP
}

// Explicit final teardown only. No automatic Python finalizer is permitted.
// graphs_destroyed is the caller's lifecycle assertion, not a HIP graph query.
extern "C" int hip_rdma_backend_close(void* handle, int graphs_destroyed,
    char* message, size_t message_bytes) {
    auto* backend = static_cast<Backend*>(handle);
    if (!backend) return 0;
    std::unique_lock<std::mutex> lock(backend->enqueue_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return error(message, message_bytes, "enqueue is active during teardown");
    if (backend->has_capture && !graphs_destroyed)
        return error(message, message_bytes, "retain backend until all captured graphs are destroyed");
#define CLOSE_HIP(call) do { const auto rc = (call); if (rc != hipSuccess) \
    return hip_error(message, message_bytes, rc, #call); } while (0)
    CLOSE_HIP(hipSetDevice(backend->device));
    CLOSE_HIP(hipDeviceSynchronize());
    backend->ready = false;
    if (backend->transport) {
        const int rc = cpu_rdma_destroy(backend->transport);
        if (rc != CPU_RDMA_OK) return error(message, message_bytes,
            cpu_rdma_last_error(backend->transport), rc);
        backend->transport = nullptr;
    }
    if (backend->descriptors) for (unsigned i = 0; i < backend->count; ++i) {
        if (backend->descriptors[i].completed) {
            CLOSE_HIP(hipEventDestroy(backend->descriptors[i].completed));
            backend->descriptors[i].completed = nullptr;
        }
    }
    if (backend->staging) { CLOSE_HIP(hipHostFree(backend->staging)); backend->staging = nullptr; }
    if (backend->arena) { CLOSE_HIP(hipHostFree(backend->arena)); backend->arena = nullptr; }
    delete[] backend->descriptors;
    lock.unlock();
    delete backend;
    return 0;
#undef CLOSE_HIP
}
