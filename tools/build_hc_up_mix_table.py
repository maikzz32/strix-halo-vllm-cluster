"""Generate the fixed-M4 HC-up/gate fusion experiment from pinned Apache-2.0 source.

Retains the original native dot product and BF16 boundary. The caller supplies
an installed-Triton sigmoid lookup table and must validate all tensor shapes.
See tests/third_party/VLLM-LICENSE and build_hc_lossless_source.py.
"""
import argparse
import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from build_hc_lossless_source import generate as generate_base, FOOTER

REMOTE = "import pathlib,subprocess,json,hashlib\ncfg=CONFIG\np=pathlib.Path(cfg['root']);p.mkdir();p.joinpath('native.hip').write_text(cfg['source'])\nc=['/opt/rocm/bin/hipcc','-std=c++17','-O3','-save-temps','--offload-arch=gfx1151','-fPIC','-shared','native.hip','-o','native.so']\nr=subprocess.run(c,cwd=p,capture_output=True,text=True,timeout=90)\nprint(json.dumps({'root':str(p),'command':c,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,\n 'binary_sha256':hashlib.sha256(p.joinpath('native.so').read_bytes()).hexdigest() if r.returncode==0 else None}),flush=True)\nraise SystemExit(r.returncode)\n"

def generate(path):
    source=generate_base(path)
    body=source.split('\nnamespace reference {\n',1)[1].split('\n}\nnamespace packed_impl',1)[0]
    start=body.index('template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,')
    end=body.index('\n#else\ntemplate <typename scalar_t, int THRDS',start)
    kernel=body[start:end]
    prefix=body[:start]
    # Prefix opens the conditional for the small-LDS kernel.
    assert kernel.count('const int _WvPrGrp, const int CuCount)')==1
    kernel=kernel.replace('wvSplitK_hf_sml_', 'hc_up_mix_').replace('const int _WvPrGrp, const int CuCount)', 'const int _WvPrGrp, const int CuCount, const scalar_t* Xn, scalar_t* Gates, const float* SigmoidTable)')
    kernel=kernel.replace('    float sum[N][YTILE] = {};','    float mixed[N][YTILE] = {};\n    for (int hc_stream=0; hc_stream<4; ++hc_stream) {\n    float sum[N][YTILE] = {};')
    old='min__(y + m, M - 1) * Kbp'
    assert kernel.count(old)==1
    kernel=kernel.replace(old,'(hc_stream * M + min__(y + m, M - 1)) * Kbp')
    old='            C[m + y + n * M] = __float2s<scalar_t>(sum[n][y]);'
    assert kernel.count(old)==1
    kernel=kernel.replace(old,'''            float rounded = __bfloat162float(__float2bfloat16(sum[n][y]));
            if (Gates) Gates[n*(M*4)+hc_stream*M+m+y]=__float2bfloat16(rounded);
            float exp_value = expf(-rounded);
            float gate_value = (1.0f / (1.0f + exp_value));
            mixed[n][y] = fmaf(gate_value, __bfloat162float(Xn[n * (M*4) + hc_stream*M + m+y]), mixed[n][y]);''')
    old='    m += CuCount * _WvPrGrp * YTILE;'
    assert kernel.count(old)==1
    kernel=kernel.replace(old,'''    } // HC streams; each retains the original dot-product and BF16 boundary.
    if (threadIdx.x == THRDS-1) {
      for (int n=0;n<N;++n) for(int y=0;y<YTILE;++y)
        C[m+y+n*M]=__float2bfloat16(mixed[n][y] * 0.25f);
    }
'''+old)
    # Distribute the eight (batch,row) epilogues across lanes 0..7.
    kernel=kernel.replace('float mixed[N][YTILE] = {};','float mixed = 0.0f;')
    a=kernel.index('      if (threadIdx.x == (THRDS - 1)) {')
    b=kernel.index('    } else {\n  #ifdef __HIP__GFX9__',a)
    kernel=kernel[:a]+'''      float selected=0.0f;
      #pragma unroll
      for (int n=0;n<N;++n) {
        #pragma unroll
        for (int y=0;y<YTILE;++y) {
          float full=__shfl(sum[n][y],THRDS-1);
          if (threadIdx.x==n*YTILE+y) selected=full;
        }
      }
      if (threadIdx.x<N*YTILE) {
        int n=threadIdx.x/YTILE, y=threadIdx.x%YTILE;
        float rounded=__bfloat162float(__float2bfloat16(selected+0.0f));
        if(Gates) Gates[n*(M*4)+hc_stream*M+m+y]=__float2bfloat16(rounded);
        union { __hip_bfloat16 value; uint16_t bits; } key;
        key.value=__float2bfloat16(rounded);
        float gate_value=SigmoidTable[key.bits];
        mixed=fmaf(gate_value,__bfloat162float(Xn[n*(M*4)+hc_stream*M+m+y]),mixed);
      }
'''+kernel[b:]
    old='''    if (threadIdx.x == THRDS-1) {
      for (int n=0;n<N;++n) for(int y=0;y<YTILE;++y)
        C[m+y+n*M]=__float2bfloat16(mixed[n][y] * 0.25f);
    }'''
    assert kernel.count(old)==1
    kernel=kernel.replace(old,'''    if(threadIdx.x<N*YTILE)
      C[m+threadIdx.x%YTILE+(threadIdx.x/YTILE)*M]=__float2bfloat16(mixed*0.25f);''')
    extra='\nnamespace fused_up {\n'+prefix+kernel+'\n#endif\n}\n'
    source=source.replace(FOOTER,extra+FOOTER)
    source+='''
extern "C" int hc_up_mix(const void* w,const void* lora,const void* xn,void* out,void* gates,const void* table,int cu,void* stream){
 if(!w||!lora||!xn||!out||cu<1||cu>128)return -1;
 hipLaunchKernelGGL((fused_up::hc_up_mix_<__hip_bfloat16,32,2,16,8,2,4>),dim3(cu),dim3(32,16),0,reinterpret_cast<hipStream_t>(stream),320,320,320,2560,1,1,static_cast<const __hip_bfloat16*>(w),static_cast<const __hip_bfloat16*>(lora),nullptr,static_cast<__hip_bfloat16*>(out),16,cu,static_cast<const __hip_bfloat16*>(xn),static_cast<__hip_bfloat16*>(gates),static_cast<const float*>(table));
 return int(hipGetLastError());
}
'''
    return source

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Compile in an isolated Node 4 container directory.')
    a=p.parse_args()
    source=generate(a.source)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'native.hip').write_bytes(source.encode())
    manifest=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  upstream_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest())
    (a.output/'source.json').write_text(json.dumps(manifest,indent=2))
    if a.execute:
        cfg={'root':'/tmp/hc-up-mix-'+uuid.uuid4().hex,'source':source}
        r=subprocess.run(['ssh','-o','BatchMode=yes','maik@192.168.1.18',
            'podman exec -i qwen029-tp4 timeout --signal=TERM --kill-after=5s 110s python3 -S -'],
            input=REMOTE.replace('CONFIG',repr(cfg),1).encode(),capture_output=True,timeout=125)
        (a.output/'build.log').write_bytes(r.stdout+r.stderr)
        print(r.stdout.decode(),r.stderr.decode(),flush=True)
        raise SystemExit(r.returncode)
    print(json.dumps(manifest))
