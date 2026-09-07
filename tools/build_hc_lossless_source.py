"""Generate isolated reference/packed HC kernels from a pinned vLLM source.

vLLM source is Apache-2.0; retain tests/third_party/VLLM-LICENSE when distributing.
The arithmetic body is identical; the packed variant changes only weight loads.
"""
import hashlib
from pathlib import Path

SOURCE_SHA = '013f14b570cd8f25e254bf47643ba2802ab7d5fdd2069adb111bc6ff560f6682'
HEADER = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <type_traits>
#define __HIP__GFX1X__
#define LDS_SIZE (64*1024)
__device__ inline float2 operator*(float2 a,float2 b){return make_float2(a.x*b.x,a.y*b.y);}
template<class T> __device__ __forceinline__ T __float2s(float x);
template<> __device__ __forceinline__ half __float2s(float x){return __float2half(x);}
template<> __device__ __forceinline__ __hip_bfloat16 __float2s(float x){return __float2bfloat16(x);}
template<class T> __device__ __forceinline__ T loadnt(T* p){return __builtin_nontemporal_load(p);}
template<class T> __device__ __forceinline__ T decode8(const uint32_t* desc,const uint8_t* payload,unsigned offset){
    unsigned d=desc[offset/256], j=offset%256;
    const uint8_t* p=payload+((d&0x7fffff)*16);
    if(d&0x800000)return loadnt(reinterpret_cast<const T*>(p+j*2));
    union {T vec;uint32_t words[4];} out;
    const uint32_t* q=reinterpret_cast<const uint32_t*>(p+j*3/2);
    uint32_t a=loadnt(q),b=loadnt(q+1),c=loadnt(q+2);
    uint32_t pairs[4]={a,(a>>24)|(b<<8),(b>>16)|(c<<16),c>>8};
    unsigned bases=((d>>24)*0x00010001u)<<7;
    #pragma unroll
    for(unsigned i=0;i<4;i++){
        unsigned h=pairs[i];
        unsigned expanded=(h&4095)|((h&0x00fff000)<<4);
        out.words[i]=((expanded&0x07ff07ff)+bases)|((expanded&0x08000800)<<4);
    }
    return out.vec;
}
'''
FOOTER = r'''
using unpack_vec = uint32_t __attribute__((vector_size(16)));
__global__ void unpack_kernel(const uint32_t* desc,const uint8_t* data,unpack_vec* out,unsigned vectors){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<vectors)out[i]=decode8<unpack_vec>(desc,data,i*8);
}
extern "C" int hc_unpack(const void* desc,const void* data,void* out,unsigned count,void* stream){
    if(!desc||!data||!out||!count||count%256)return -1;
    hipLaunchKernelGGL(unpack_kernel,dim3((count/8+255)/256),dim3(256),0,reinterpret_cast<hipStream_t>(stream),static_cast<const uint32_t*>(desc),static_cast<const uint8_t*>(data),static_cast<unpack_vec*>(out),count/8);
    return int(hipGetLastError());
}
extern "C" int hc_run(int packed,const void* w,const void* x,void* out,int rows,int k,int cu,void* stream,const void* desc,const void* data){
    if(!x||!out||cu<1||cu>128||!w)return -1;
    if(!((rows==336&&k==10240)||(rows==320&&k==10240)||(rows==10240&&k==320)))return -2;
    if(packed&&(!desc||!data))return -3;
    auto s=reinterpret_cast<hipStream_t>(stream);
    auto a=static_cast<const __hip_bfloat16*>(x),b=static_cast<const __hip_bfloat16*>(w);
    auto c=static_cast<__hip_bfloat16*>(out);
    dim3 grid(cu),block(32,16);
    if(packed==3||packed==4){
        if(k==320){
            int waves=reference::mindiv(rows,cu*2,16);
            if(packed==3)hipLaunchKernelGGL((reference::wvSplitK_hf_sml_<__hip_bfloat16,32,2,16,8,1,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu);
            else hipLaunchKernelGGL((reference::wvSplitK_hf_sml_<__hip_bfloat16,32,2,16,8,2,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu);
        }else{
            int waves=reference::mindiv(rows,cu,16);
            if(packed==3)hipLaunchKernelGGL((reference::wvSplitK_hf_big_<__hip_bfloat16,32,1,16,8,1,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu);
            else hipLaunchKernelGGL((reference::wvSplitK_hf_big_<__hip_bfloat16,32,1,16,8,2,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu);
        }
        return int(hipGetLastError());
    }
    if(k==320){
        int waves=reference::mindiv(rows,cu*2,16);
        if(packed)hipLaunchKernelGGL((packed_impl::wvSplitK_hf_sml_<__hip_bfloat16,32,2,16,8,2,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu,static_cast<const uint32_t*>(desc),static_cast<const uint8_t*>(data));
        else hipLaunchKernelGGL((reference::wvSplitK_hf_sml_<__hip_bfloat16,32,2,16,8,4,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu);
    }else{
        int waves=(packed==2 && rows==336)?8:reference::mindiv(rows,cu,16);
        if(packed)hipLaunchKernelGGL((packed_impl::wvSplitK_hf_big_<__hip_bfloat16,32,1,16,8,2,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu,static_cast<const uint32_t*>(desc),static_cast<const uint8_t*>(data));
        else hipLaunchKernelGGL((reference::wvSplitK_hf_big_<__hip_bfloat16,32,1,16,8,4,4>),grid,block,0,s,k,k,k,rows,1,1,b,a,nullptr,c,waves,cu);
    }
    return int(hipGetLastError());
}
'''


def generate(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('unreviewed upstream source')
    source = raw.decode()
    start = source.index('#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)')
    end = source.index('torch::Tensor wvSplitK(')
    body = source[start:end]
    old = 'const int _WvPrGrp, const int CuCount)'
    assert body.count(old) == 6
    changed = body.replace(old, old[:-1] + ', const uint32_t* desc, const uint8_t* payload)')
    pointer = 'const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];'
    load = 'bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));'
    assert changed.count(pointer) == changed.count(load) == 3
    changed = changed.replace(pointer, '').replace(load,
        'bigB[y][k2].h8 = decode8<scalar8>(desc,payload,min__(k_, K-A_CHUNK)+min__(y+m,M-1)*Kbp);')
    return ('// Derived from vLLM 33898f832; Apache-2.0, see VLLM-LICENSE.\n' + HEADER +
            '\nnamespace reference {\n' + body + '\n}\nnamespace packed_impl {\n' + changed + '\n}\n' + FOOTER)
