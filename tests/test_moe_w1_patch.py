"""CPU-only checks for the opt-in W1 source patch and its shape boundary."""
import argparse, ast, hashlib, json, sys, types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from patch_moe_w1 import patched

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);args=p.parse_args()
 base=args.source.read_bytes();new=patched(base)
 try:patched(new)
 except ValueError:pass
 else:raise AssertionError('Must reject already modified source')
 tree=ast.parse(new);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_glm53_moe_int4_gemv')
 guards=[n for n in ast.walk(fn) if isinstance(n,ast.If) and 'STRIX_MOE_W1_BN32' in ast.unparse(n.test)]
 assert len(guards)==1
 code=compile(ast.fix_missing_locations(ast.Module(body=[guards[0]],type_ignores=[])),'shape-guard','exec')
 def check(env='1',**changes):
  ns=dict(os=types.SimpleNamespace(environ={'STRIX_MOE_W1_BN32':env}),M=4,top_k=10,N=320,K=2560,GS=32,config={'BLOCK_SIZE_M':16},BN=64,num_warps=4);ns.update(changes);exec(code,ns);return ns['BN'],ns['num_warps']
 assert check()==(32,2)
 assert check('0')==(64,4)
 for k,v in [('M',1),('M',8),('top_k',1),('N',2560),('K',160),('GS',64),('config',{'BLOCK_SIZE_M':32})]:assert check(**{k:v})==(64,4)
 print(json.dumps({'candidate_sha256':hashlib.sha256(new).hexdigest(),'guard_cases_passed':9,'rejects_modified_source':True}))
if __name__=='__main__':main()
