"""Same-split W1 occupancy experiment using the existing independent-reference harness."""
import argparse, contextlib, io, json, hashlib
from pathlib import Path
import bench_moe_w1_geometry as study

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 source=study.DEFAULT_SOURCE.read_text(encoding='utf-8')
 assert hashlib.sha256(source.encode()).hexdigest()=='4f460bcc4c6c074dc8c08c3881bccdbdbd941490821bac458c9782c1404a0070'
 study.GEOMETRIES=[(4,128,64,4,2),(4,128,64,2,2),(4,128,32,2,2),(4,128,64,4,1)]
 output=io.StringIO()
 try:
  with contextlib.redirect_stdout(output):study.gpu_study(source,120)
 finally:
  lines=output.getvalue().splitlines();rows=[json.loads(x) for x in lines if x.startswith('{')]
  report={'status':'passed' if rows and rows[-1].get('gpu_released_on_process_exit') else 'failed','events':rows}
  args.output.write_text(json.dumps(report,indent=2));print(output.getvalue(),flush=True)
if __name__=='__main__':main()
