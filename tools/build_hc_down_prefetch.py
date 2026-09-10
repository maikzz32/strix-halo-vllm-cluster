"""Generate an isolated HC-down weight-prefetch experiment from pinned source."""
import argparse
import hashlib
from pathlib import Path
from build_hc_lossless_source import HEADER, SOURCE_SHA


def generate(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    source = raw.decode()
    body = source[source.index('#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)'):source.index('torch::Tensor wvSplitK(')]
    start = body.index('    wvSplitK_hf_big_(')
    end = body.index('\n#else', body.index('    m +=', start))
    kernel = body[start:end]
    loop = '    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {'
    assert kernel.count(loop) == 1
    initial = '''    bigType nextB[YTILE][UNRL];
    if (m < M) {
      #pragma unroll
      for (uint32_t k2=0; k2<UNRL; ++k2) {
        uint32_t offset=k2*THRDS*A_CHUNK+threadIdx.x*A_CHUNK;
        for (int y=0; y<YTILE; ++y)
          nextB[y][k2].h8=loadnt((scalar8*)(&B[min__(offset,K-A_CHUNK)+min__(y+m,M-1)*Kbp]));
      }
    }
'''
    kernel = kernel.replace(loop, initial + loop)
    load_start = kernel.index('      // Fetch the weight matrix from memory!', kernel.index(loop))
    load_end = kernel.index('      // Fetch activation', load_start)
    kernel = kernel[:load_start] + '''      #pragma unroll
      for (uint32_t k2=0; k2<UNRL; ++k2)
        for (int y=0; y<YTILE; ++y) bigB[y][k2]=nextB[y][k2];

''' + kernel[load_end:]
    marker = '      // Do the matrix multiplication'
    assert kernel.count(marker) == 1
    preload = '''      // Fetch only the next weight tile; dot-product order below is unchanged.
      if (k1 + THRDS*A_CHUNK*UNRL < K) {
        #pragma unroll
        for (uint32_t k2=0; k2<UNRL; ++k2) {
          uint32_t offset=k1+THRDS*A_CHUNK*UNRL+k2*THRDS*A_CHUNK+threadIdx.x*A_CHUNK;
          for (int y=0; y<YTILE; ++y)
            nextB[y][k2].h8=loadnt((scalar8*)(&B[min__(offset,K-A_CHUNK)+min__(y+m,M-1)*Kbp]));
        }
      }
'''
    kernel = kernel.replace(marker, preload + marker)
    candidate = body[:start] + kernel + body[end:]
    footer = r'''
extern "C" int hc_down_prefetch(int mode,const void* w,const void* x,void* out,int rows,int cu,void* stream){
  if(mode<0||mode>1||!w||!x||!out||(rows!=320&&rows!=336)||cu!=20)return -1;
  auto s=reinterpret_cast<hipStream_t>(stream);
  auto b=static_cast<const __hip_bfloat16*>(w),a=static_cast<const __hip_bfloat16*>(x);
  auto c=static_cast<__hip_bfloat16*>(out);
  int waves=reference::mindiv(rows,cu,16);
  dim3 grid(cu),block(32,16);
  if(mode==0)hipLaunchKernelGGL((reference::wvSplitK_hf_big_<__hip_bfloat16,32,1,16,8,4,4>),grid,block,0,s,10240,10240,10240,rows,1,1,b,a,nullptr,c,waves,cu);
  else hipLaunchKernelGGL((prefetch::wvSplitK_hf_big_<__hip_bfloat16,32,1,16,8,4,4>),grid,block,0,s,10240,10240,10240,rows,1,1,b,a,nullptr,c,waves,cu);
  return int(hipGetLastError());
}
'''
    return '// Derived from vLLM 33898f832, Apache-2.0; see tests/third_party/VLLM-LICENSE.\n'+HEADER+'\nnamespace reference {\n'+body+'\n}\nnamespace prefetch {\n'+candidate+'\n}\n'+footer


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=generate(a.source);Path(a.output).write_text(out)
    print(hashlib.sha256(out.encode()).hexdigest())
