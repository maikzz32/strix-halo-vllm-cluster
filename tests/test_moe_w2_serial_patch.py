"""Evaluate actual candidate dispatch guard with CPU tensor metadata doubles."""
import argparse,ast,hashlib,json,sys,types,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from patch_moe_w2_serial import patched

class Tensor:
 def __init__(self,shape,dtype,contiguous=True):self.shape=shape;self.dtype=dtype;self.contiguous=contiguous
 def size(self,i):return self.shape[i]
 def numel(self):return math.prod(self.shape)
 def is_contiguous(self):return self.contiguous

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 base=a.source.read_bytes();candidate=patched(base)
 for wrong in (candidate,base+b'\n'):
  try:patched(wrong)
  except ValueError:pass
  else:raise AssertionError('Modified source accepted')
 fn=next(n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name=='_glm53_moe_int4_gemv')
 guards=[n for n in ast.walk(fn) if isinstance(n,ast.If) and 'STRIX_MOE_W2_SERIAL' in ast.unparse(n.test)]
 assert len(guards)==1
 expr=compile(ast.Expression(guards[0].test),'actual-w2-guard','eval')
 ns=dict(os=types.SimpleNamespace(environ={'STRIX_MOE_W2_SERIAL':'1'}),torch=types.SimpleNamespace(float32='f32',bfloat16='bf16',uint8='u8'),M=40,top_k=1,N=2560,K=160,GS=32,BM=16,mul_routed_weight=True,A=Tensor((40,160),'bf16'),C=Tensor((4,10,2560),'bf16'),B=Tensor((512,2560,80),'u8'),B_scale=Tensor((512,2560,5),'bf16'),B_zp=Tensor((512,1280,5),'u8'),topk_weights=Tensor((4,10),'f32'))
 assert eval(expr,ns)
 rejected=[]
 for key,value in [('M',1),('M',80),('top_k',10),('N',320),('K',640),('GS',64),('BM',32),('mul_routed_weight',False),('topk_weights',None),('topk_weights',Tensor((4,9),'f32')),('topk_weights',Tensor((4,10),'bf16')),('topk_weights',Tensor((4,10),'f32',False)),('A',Tensor((40,160),'f32')),('C',Tensor((4,10,2560),'f32')),('B',Tensor((128,2560,80),'u8')),('B',Tensor((512,2560,80),'u8',False)),('B_scale',Tensor((512,2560,5),'f32')),('B_scale',Tensor((512,2560,4),'bf16')),('B_scale',Tensor((512,2560,5),'bf16',False)),('B_zp',Tensor((512,1280,5),'f32')),('B_zp',Tensor((512,1280,4),'u8')),('B_zp',Tensor((512,1280,5),'u8',False))]:
  trial=dict(ns);trial[key]=value;assert not eval(expr,trial),key;rejected.append(key)
 for env in ({},{'STRIX_MOE_W2_SERIAL':'0'}):
  trial=dict(ns);trial['os']=types.SimpleNamespace(environ=env);assert not eval(expr,trial)
 print(json.dumps({'candidate_sha256':hashlib.sha256(candidate).hexdigest(),'accepted':1,'rejected':len(rejected)+2,'modified_source_rejected':2}))
if __name__=='__main__':main()
