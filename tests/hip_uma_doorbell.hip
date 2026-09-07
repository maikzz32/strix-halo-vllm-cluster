// Isolated GPU/CPU mapped-memory handshake, no model hook or network access.
// Each graph replay advances a GPU-owned epoch; host-enqueue counters are absent.
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
#include <immintrin.h>

constexpr unsigned N=10240, OPS=32;
struct alignas(64) Shared {
    alignas(64) uint64_t request;
    alignas(64) uint64_t completed;
    alignas(64) uint32_t error;
    alignas(64) uint16_t input[N];
    alignas(64) uint16_t output[N];
};
__host__ __device__ uint16_t pattern(unsigned i,uint64_t epoch) {
    if (i<4) return uint16_t(epoch>>(16*i)); // Full epoch, no 256-replay alias.
    return uint16_t(0x3f00u + ((i+17*epoch)%256));
}
__global__ void exchange(Shared* s,uint64_t* counter,uint16_t* out) {
    __shared__ uint64_t epoch;
    __shared__ unsigned failed;
    if (threadIdx.x==0) { epoch=++*counter; failed=0; }
    __syncthreads();
    for (unsigned i=threadIdx.x;i<N;i+=blockDim.x) s->input[i]=pattern(i,epoch);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x==0) {
        __hip_atomic_store(&s->request,epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
        bool ready=false;
        for (unsigned spin=0;spin<10000000;spin++) {
            if (__hip_atomic_load(&s->error,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)) {
                failed=1;break;
            }
            auto done=__hip_atomic_load(&s->completed,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
            if (done==epoch) {ready=true;break;}
            if (done>epoch) {failed=1;break;}
            __builtin_amdgcn_s_sleep(1);
        }
        if (!ready) {
            failed=1;
            __hip_atomic_store(&s->error,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    __syncthreads();
    // A failed exchange leaves the distinct caller output entirely untouched.
    if (!failed) for (unsigned i=threadIdx.x;i<N;i+=blockDim.x) out[i]=s->output[i];
}
void check(hipError_t rc,const char* call) {
    if (rc!=hipSuccess) {std::fprintf(stderr,"%s: %s\n",call,hipGetErrorString(rc));std::_Exit(2);}
}
#define HIP(call) check(call,#call)
void require(bool ok,const char* message) {
    if (!ok) {std::fprintf(stderr,"%s\n",message);std::_Exit(3);}
}
int main() {
    Shared *host=nullptr,*device=nullptr;
    HIP(hipHostMalloc(reinterpret_cast<void**>(&host),sizeof(Shared),hipHostMallocMapped));
    std::memset(host,0,sizeof(Shared));
    HIP(hipHostGetDevicePointer(reinterpret_cast<void**>(&device),host,0));
    uint64_t* counter=nullptr;uint16_t* output=nullptr;
    HIP(hipMalloc(&counter,sizeof(uint64_t)));HIP(hipMemset(counter,0,sizeof(uint64_t)));
    HIP(hipMalloc(&output,OPS*N*sizeof(uint16_t)));
    hipStream_t stream;HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::atomic<bool> stop{false};std::atomic<uint64_t> fail_at{0};
    uint64_t calls=0,bad=0;
    std::thread progress([&] {
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while (!stop.load(std::memory_order_acquire)) {
            auto epoch=__atomic_load_n(&host->request,__ATOMIC_ACQUIRE);
            if (epoch==calls) {
                if (std::chrono::steady_clock::now()>deadline) {
                    __atomic_store_n(&host->error,2u,__ATOMIC_RELEASE);break;
                }
                _mm_pause();continue;
            }
            if (epoch!=calls+1 || epoch==fail_at.load(std::memory_order_acquire)) {
                __atomic_store_n(&host->error,3u,__ATOMIC_RELEASE);break;
            }
            for (unsigned i=0;i<N;i++) {
                bad+=host->input[i]!=pattern(i,epoch);
                host->output[i]=uint16_t(host->input[i]^0x8000u);
            }
            if (bad) {__atomic_store_n(&host->error,4u,__ATOMIC_RELEASE);break;}
            calls=epoch;
            __atomic_store_n(&host->completed,epoch,__ATOMIC_RELEASE);
            deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        }
    });
    uint64_t expected=0;
    std::vector<uint16_t> copied(OPS*N);
    auto launch=[&](unsigned index) {
        hipLaunchKernelGGL(exchange,dim3(1),dim3(256),0,stream,device,counter,output+index*N);
        HIP(hipGetLastError());
    };
    auto verify=[&](unsigned count) {
        HIP(hipStreamSynchronize(stream));
        require(__atomic_load_n(&host->error,__ATOMIC_ACQUIRE)==0,"exchange error");
        HIP(hipMemcpy(copied.data(),output,count*N*sizeof(uint16_t),hipMemcpyDeviceToHost));
        for (unsigned op=0;op<count;op++) for (unsigned i=0;i<N;i++)
            require(copied[op*N+i]==uint16_t(pattern(i,expected-count+op+1)^0x8000u),"stale or incorrect GPU result");
        uint64_t actual=0;HIP(hipMemcpy(&actual,counter,sizeof(actual),hipMemcpyDeviceToHost));
        require(actual==expected,"GPU graph epoch did not advance");
    };
    for (unsigned i=0;i<4;i++) {launch(0);expected++;verify(1);}
    hipGraph_t graph;hipGraphExec_t exec;
    HIP(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
    for (unsigned i=0;i<OPS;i++) launch(i);
    HIP(hipStreamEndCapture(stream,&graph));
    size_t nodes=0;HIP(hipGraphGetNodes(graph,nullptr,&nodes));
    require(nodes==OPS,"graph must contain only the expected kernel nodes");
    HIP(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (unsigned i=0;i<4;i++) {HIP(hipGraphLaunch(exec,stream));expected+=OPS;verify(OPS);}
    hipEvent_t start,end;HIP(hipEventCreate(&start));HIP(hipEventCreate(&end));
    std::vector<double> times;
    for (unsigned sample=0;sample<3;sample++) {
        HIP(hipEventRecord(start,stream));
        for (unsigned replay=0;replay<32;replay++) {HIP(hipGraphLaunch(exec,stream));expected+=OPS;}
        HIP(hipEventRecord(end,stream));HIP(hipEventSynchronize(end));
        float ms=0;HIP(hipEventElapsedTime(&ms,start,end));times.push_back(ms*1000/(32*OPS));verify(OPS);
    }
    // Fault injection after successful graph replays; no stale result may escape.
    fail_at.store(expected+1,std::memory_order_release);
    HIP(hipMemsetAsync(output,0xa5,N*sizeof(uint16_t),stream));launch(0);
    HIP(hipStreamSynchronize(stream));
    require(__atomic_load_n(&host->error,__ATOMIC_ACQUIRE)!=0,"injected failure was ignored");
    HIP(hipMemcpy(copied.data(),output,N*sizeof(uint16_t),hipMemcpyDeviceToHost));
    for(unsigned i=0;i<N;i++)require(copied[i]==0xa5a5,"failure published stale output");
    stop.store(true,std::memory_order_release);progress.join();
    require(calls==expected && bad==0,"host sequence or input validation failed");
    HIP(hipGraphExecDestroy(exec));HIP(hipGraphDestroy(graph));
    HIP(hipEventDestroy(start));HIP(hipEventDestroy(end));HIP(hipStreamDestroy(stream));
    HIP(hipFree(output));HIP(hipFree(counter));HIP(hipHostFree(host));
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    std::printf("{\"status\":\"passed\",\"successful_epochs\":%llu,\"input_values_checked\":%llu,\"graph_kernel_nodes\":%zu,\"samples_us\":[%.6f,%.6f,%.6f],\"median_us\":%.6f,\"fault_output_untouched\":true,\"resources_released\":true}\n",
        (unsigned long long)calls,(unsigned long long)(calls*N),nodes,times[0],times[1],times[2],sorted[1]);
}
