"""Package pinned UCCL host-plugin sources without installing or replacing RCCL."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
UCCL='79b64ae7ca58ea78bb39c6275b0f91443e33a3f2'
RCCL='532f54c2444501b3655e65fbce6d00d4bfc19c0b'

def git(root,*args):return subprocess.run(['git','-C',str(root),*args],capture_output=True,check=True).stdout

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert git(a.source,'rev-parse','HEAD').decode().strip()==UCCL
 rccl=a.source/'thirdparty/rccl';assert git(rccl,'rev-parse','HEAD').decode().strip()==RCCL
 files={}
 for root,paths,prefix in [(a.source,['collective/rdma','include','LICENSE'],''),(rccl,['src/include','LICENSE.txt'],'rccl/')]:
  raw=git(root,'archive','HEAD',*paths)
  with tarfile.open(fileobj=io.BytesIO(raw)) as ar:
   for entry in ar:
    if entry.isfile():files[prefix+entry.name]=ar.extractfile(entry).read()
 text=git(rccl,'show','HEAD:src/nccl.h.in').decode()
 versions={'NCCL_MAJOR':'2','NCCL_MINOR':'23','NCCL_PATCH':'4','NCCL_SUFFIX':'','NCCL_VERSION':'22304'}
 for key,value in versions.items():text=text.replace('${'+key+'}',value)
 assert '${' not in text
 files['rccl/src/include/nccl.h']=text.encode()
 files['build-plugin.sh']=b'''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
mkdir build
flags=(-O3 -g -std=c++17 -fPIC -D__HIP_PLATFORM_AMD__ -DINTEL_RDMA_NIC -Wno-pointer-arith -Wno-interference-size -Icollective/rdma -Iinclude -Irccl/src/include -I/opt/rocm/include)
# Host C++ plugin only. Keep debug assertions; do not claim non-GDR support.
for name in transport rdma_io param eqds nccl_plugin; do
  g++ "${flags[@]}" -c "collective/rdma/$name.cc" -o "build/$name.o"
done
# gflags/gtest are used by separate test programs, not these plugin sources.
g++ -shared -Wl,-z,defs -Wl,-soname,librccl-net-uccl.so -o build/librccl-net-uccl.so build/*.o -libverbs -lpthread -ldl -L/opt/rocm/lib -lamdhip64
sha256sum build/librccl-net-uccl.so
nm -D build/librccl-net-uccl.so | grep ncclNetPlugin
'''
 manifest={'uccl_commit':UCCL,'rccl_header_commit':RCCL,'generated_header_versions':versions,'files':{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())},'scope':'Build inputs only; host C++ plugin, Intel macro enabled, debug assertions retained, no RCCL replacement or activation.'}
 a.output.mkdir(parents=True,exist_ok=False)
 with tarfile.open(a.output/'source.tar.gz','w:gz') as ar:
  for name,content in files.items():
   t=tarfile.TarInfo(name);t.size=len(content);t.mode=0o755 if name.endswith('.sh') else 0o644;ar.addfile(t,io.BytesIO(content))
 manifest['archive_sha256']=hashlib.sha256((a.output/'source.tar.gz').read_bytes()).hexdigest()
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'file_count':len(files),'archive_sha256':manifest['archive_sha256']}))
if __name__=='__main__':main()
