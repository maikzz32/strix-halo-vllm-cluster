// Experimental TP4 transport: one GPU kernel, mapped UMA arena, CPU verbs thread.
// Same C ABI as hip_rdma_backend.cpp. No serving installation on import/build.
#include "cpu_rdma_transport.h"
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <immintrin.h>
#include <unistd.h>
#ifndef STRIX_UMA_THREADS
#define STRIX_UMA_THREADS 256
#endif
static_assert(STRIX_UMA_THREADS==256||STRIX_UMA_THREADS==512||STRIX_UMA_THREADS==1024);

namespace {
struct alignas(64) Control {
    alignas(64) uint64_t request;
    alignas(64) uint64_t completed;
    alignas(64) uint32_t error;
    alignas(64) uint32_t active;
};
struct Backend {
    cpu_rdma_ctx* transport=nullptr;
    void *arena=nullptr,*arena_device=nullptr;
    Control *control=nullptr,*control_device=nullptr;
    uint64_t* counter=nullptr;
    int device=-1,rank=-1;
    bool ready=false,has_capture=false;
    std::atomic<bool> stop{false},thread_ready{false};
    std::thread progress;
    std::mutex mutex;
};
int error(char* message,size_t size,const char* text,int code=-1) {
    if(message&&size)std::snprintf(message,size,"%s",text);return code;
}
[[noreturn]] void fail(Backend* b,const char* text) {
    char line[768];int n=std::snprintf(line,sizeof(line),"HIP_UMA_FATAL rank=%d: %s\n",b->rank,text);
    if(n>0)(void)!write(2,line,size_t(n)<sizeof(line)?size_t(n):sizeof(line)-1);
    _exit(86);
}
__global__ void exchange(const uint16_t* input,uint16_t* output,uint16_t* staged,
                         const uint16_t* reduced,Control* control,uint64_t* counter) {
    __shared__ uint64_t epoch;
    __shared__ unsigned failed;
    if(threadIdx.x==0) {
        failed=__hip_atomic_exchange(&control->active,1u,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_SYSTEM);
        if(!failed)epoch=++*counter;
        else __hip_atomic_store(&control->error,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    if(failed)return;
    for(unsigned i=threadIdx.x;i<CPU_RDMA_VALUES;i+=blockDim.x)staged[i]=input[i];
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0) {
        __hip_atomic_store(&control->request,epoch,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
        bool ready=false;
        for(unsigned spin=0;spin<10000000;spin++) {
            if(__hip_atomic_load(&control->error,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM))break;
            auto completed=__hip_atomic_load(&control->completed,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
            if(completed==epoch){ready=true;break;}
            if(completed>epoch)break;
            __builtin_amdgcn_s_sleep(1);
        }
        if(!ready) {
            failed=1;
            __hip_atomic_store(&control->error,2u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    __syncthreads();
    if(!failed)for(unsigned i=threadIdx.x;i<CPU_RDMA_VALUES;i+=blockDim.x)output[i]=reduced[i];
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0)__hip_atomic_store(&control->active,0u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
}
void progress(Backend* b) {
    char message[512]={};
    if(cpu_rdma_numerics_selftest(message,sizeof(message))!=CPU_RDMA_OK)fail(b,message);
    uint64_t completed=0;
    auto* input=static_cast<char*>(b->arena)+CPU_RDMA_INPUT_OFFSET;
    auto* output=static_cast<char*>(b->arena)+CPU_RDMA_OUTPUT_OFFSET;
    b->thread_ready.store(true,std::memory_order_release);
    while(!b->stop.load(std::memory_order_acquire)) {
        if(__atomic_load_n(&b->control->error,__ATOMIC_ACQUIRE))fail(b,"GPU exchange failed");
        auto request=__atomic_load_n(&b->control->request,__ATOMIC_ACQUIRE);
        if(request==completed){_mm_pause();continue;}
        if(request!=completed+1)fail(b,"non-sequential GPU epoch");
        if(cpu_rdma_run(b->transport,input,output,CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16)!=CPU_RDMA_OK)
            fail(b,cpu_rdma_last_error(b->transport));
        completed=request;
        __atomic_store_n(&b->control->completed,completed,__ATOMIC_RELEASE);
    }
}
}
extern "C" unsigned hip_rdma_backend_abi_version(){return 1;}
extern "C" size_t hip_rdma_backend_params_size(){return sizeof(cpu_rdma_params);}
extern "C" int hip_rdma_backend_create(const cpu_rdma_params* params,int device,int mode,
    unsigned slots,void** out,char* message,size_t size) {
    if(out)*out=nullptr;
    if(!params||!out||device<0||mode!=2||slots<128||slots>8192)return error(message,size,"invalid UMA options");
    auto* b=new(std::nothrow)Backend;if(!b)return error(message,size,"allocation failed");
    *out=b;b->device=device;b->rank=int(params->rank);
#define H(call) do{auto rc=(call);if(rc!=hipSuccess)return error(message,size,hipGetErrorString(rc),-1000-int(rc));}while(0)
    H(hipSetDevice(device));
    hipStreamCaptureStatus capture;H(hipStreamIsCapturing(nullptr,&capture));
    if(capture!=hipStreamCaptureStatusNone)return error(message,size,"create during capture");
    H(hipHostMalloc(&b->arena,CPU_RDMA_ARENA_BYTES,hipHostMallocMapped));
    H(hipHostGetDevicePointer(&b->arena_device,b->arena,0));
    H(hipHostMalloc(reinterpret_cast<void**>(&b->control),sizeof(Control),hipHostMallocMapped));
    std::memset(b->control,0,sizeof(Control));
    H(hipHostGetDevicePointer(reinterpret_cast<void**>(&b->control_device),b->control,0));
    H(hipMalloc(&b->counter,sizeof(uint64_t)));H(hipMemset(b->counter,0,sizeof(uint64_t)));
    int rc=cpu_rdma_create(params,b->arena,CPU_RDMA_ARENA_BYTES,&b->transport,message,size);
    if(rc)return rc;
    try{b->progress=std::thread(progress,b);}catch(...){return error(message,size,"progress thread creation failed");}
    while(!b->thread_ready.load(std::memory_order_acquire))std::this_thread::yield();
    b->ready=true;return 0;
#undef H
}
extern "C" int hip_rdma_backend_enqueue(void* handle,const void* input,void* output,
    size_t count,void* stream_pointer,char* message,size_t size) {
    auto* b=static_cast<Backend*>(handle);
    if(!b||!b->ready||!input||!output||count!=CPU_RDMA_VALUES)return error(message,size,"invalid UMA enqueue");
    auto a=reinterpret_cast<uintptr_t>(input),c=reinterpret_cast<uintptr_t>(output);
    if((a>=c?a-c:c-a)<CPU_RDMA_BYTES)return error(message,size,"output must not overlap input");
    std::unique_lock<std::mutex> lock(b->mutex,std::try_to_lock);
    if(!lock.owns_lock())return error(message,size,"concurrent enqueue");
    int current=-1;auto rc=hipGetDevice(&current);
    if(rc!=hipSuccess||current!=b->device)return error(message,size,"incorrect HIP device");
    auto stream=reinterpret_cast<hipStream_t>(stream_pointer);
    hipStreamCaptureStatus capture;rc=hipStreamIsCapturing(stream,&capture);
    if(rc!=hipSuccess||capture==hipStreamCaptureStatusInvalidated)return error(message,size,"invalid capture");
    b->has_capture|=capture!=hipStreamCaptureStatusNone;
    hipLaunchKernelGGL(exchange,dim3(1),dim3(STRIX_UMA_THREADS),0,stream,
        static_cast<const uint16_t*>(input),static_cast<uint16_t*>(output),
        reinterpret_cast<uint16_t*>(static_cast<char*>(b->arena_device)+CPU_RDMA_INPUT_OFFSET),
        reinterpret_cast<uint16_t*>(static_cast<char*>(b->arena_device)+CPU_RDMA_OUTPUT_OFFSET),b->control_device,b->counter);
    rc=hipGetLastError();if(rc!=hipSuccess)fail(b,hipGetErrorString(rc));
    return 0;
}
extern "C" int hip_rdma_backend_close(void* handle,int graphs_destroyed,char* message,size_t size) {
    auto* b=static_cast<Backend*>(handle);if(!b)return 0;
    std::unique_lock<std::mutex> lock(b->mutex,std::try_to_lock);
    if(!lock.owns_lock()||(b->has_capture&&!graphs_destroyed))return error(message,size,"retain live backend");
#define H(call) do{auto rc=(call);if(rc!=hipSuccess)return error(message,size,hipGetErrorString(rc));}while(0)
    H(hipSetDevice(b->device));H(hipDeviceSynchronize());b->ready=false;
    b->stop.store(true,std::memory_order_release);if(b->progress.joinable())b->progress.join();
    if(b->transport){int rc=cpu_rdma_destroy(b->transport);if(rc)return error(message,size,"transport cleanup failed",rc);b->transport=nullptr;}
    if(b->counter){H(hipFree(b->counter));b->counter=nullptr;}
    if(b->arena){H(hipHostFree(b->arena));b->arena=nullptr;}
    if(b->control){H(hipHostFree(b->control));b->control=nullptr;}
    lock.unlock();delete b;return 0;
#undef H
}
