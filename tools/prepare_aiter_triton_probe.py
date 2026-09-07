"""Package pinned AITER Python/config sources for an isolated Triton import probe."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
PIN='e8eac17259db6efcb48eba2aa0d9a3bf422e7b35'
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 def git(*args):return subprocess.check_output(['git','-C',str(a.source),*args])
 assert git('rev-parse','HEAD').decode().strip()==PIN
 assert not git('status','--porcelain')
 entries=git('ls-tree','-rz',PIN,'--','aiter','LICENSE','requirements.txt').decode().split('\0')
 blobs={}
 for entry in filter(None,entries):
  meta,name=entry.split('\t',1);mode,kind,sha=meta.split()
  if kind=='blob' and (Path(name).suffix in ('.py','.json','.csv') or name in ('LICENSE','requirements.txt')):blobs[name]=sha
 paths=list(blobs)
 batch=subprocess.run(['git','-C',str(a.source),'cat-file','--batch'],input=('\n'.join(blobs.values())+'\n').encode(),capture_output=True,check=True).stdout
 stream=io.BytesIO(batch)
 a.output.mkdir(parents=True,exist_ok=False);manifest={'commit':PIN,'files':{},'scope':'Python/config sources only; no package installation or compiled binaries'}
 with tarfile.open(a.output/'sources.tar.gz','w:gz') as t:
  for name in paths:
   sha,kind,size=stream.readline().decode().split();assert sha==blobs[name] and kind=='blob'
   data=stream.read(int(size));assert stream.read(1)==b'\n'
   info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
   t.addfile(info,io.BytesIO(data));manifest['files'][name]=hashlib.sha256(data).hexdigest()
 manifest['archive_sha256']=hashlib.sha256((a.output/'sources.tar.gz').read_bytes()).hexdigest()
 (a.output/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
 print(json.dumps({'commit':PIN,'files':len(paths),'bytes':(a.output/'sources.tar.gz').stat().st_size,'sha256':manifest['archive_sha256']}))
if __name__=='__main__':main()
