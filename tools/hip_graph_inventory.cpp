// Read-only graph topology capture at instantiation. No replay interception.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <unordered_map>

static void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error("HIP graph query failed"); }
static void graph_json(FILE* f, hipGraph_t g, unsigned depth) {
  if(depth>8) throw std::runtime_error("nested graph depth exceeded");
  size_t count=0; check(hipGraphGetNodes(g,nullptr,&count));
  if(count>100000) throw std::runtime_error("node bound exceeded");
  std::vector<hipGraphNode_t> nodes(count);
  check(hipGraphGetNodes(g,nodes.data(),&count));
  std::unordered_map<hipGraphNode_t,size_t> indices;
  for(size_t i=0;i<count;++i) indices[nodes[i]]=i;
  fprintf(f,"{\"node_count\":%zu,\"nodes\":[",count);
  for(size_t i=0;i<count;++i) {
    hipGraphNodeType type;check(hipGraphNodeGetType(nodes[i],&type));
    size_t n=0;check(hipGraphNodeGetDependencies(nodes[i],nullptr,&n));
    std::vector<hipGraphNode_t> deps(n);
    if(n)check(hipGraphNodeGetDependencies(nodes[i],deps.data(),&n));
    fprintf(f,"%s{\"index\":%zu,\"type\":%d,\"dependencies\":[",i?",":"",i,int(type));
    for(size_t j=0;j<n;++j)fprintf(f,"%s%zu",j?",":"",indices.at(deps[j]));
    fprintf(f,"]");
    if(type==hipGraphNodeTypeMemcpy) {
      hipMemcpy3DParms p{};check(hipGraphMemcpyNodeGetParams(nodes[i],&p));
      fprintf(f,",\"copy\":{\"kind\":%d,\"width\":%zu,\"height\":%zu,\"depth\":%zu}",int(p.kind),p.extent.width,p.extent.height,p.extent.depth);
    }
    if(type==hipGraphNodeTypeGraph) {
      hipGraph_t child;check(hipGraphChildGraphNodeGetGraph(nodes[i],&child));
      fprintf(f,",\"child\":");graph_json(f,child,depth+1);
    }
    fprintf(f,"}");
  }
  fprintf(f,"]}");
}
static void inventory(hipGraph_t graph) noexcept {
  const char* dir=getenv("STRIX_GRAPH_INVENTORY_DIR");if(!dir||!*dir)return;
  static std::atomic<unsigned> serial{0};unsigned id=serial++;
  char path[4096];int len=snprintf(path,sizeof(path),"%s/graph-%d-%u.json",dir,int(getpid()),id);
  if(len<0||size_t(len)>=sizeof(path))return;
  FILE* f=fopen(path,"wx");if(!f)return;
  try {graph_json(f,graph,0);fprintf(f,"\n");fclose(f);}
  catch(...) {fclose(f);remove(path);fprintf(stderr,"STRIX graph inventory failed pid=%d graph=%u\n",int(getpid()),id);}
}
// PyTorch may use either API depending on the build. Delegate arguments unchanged.
extern "C" hipError_t hipGraphInstantiate(hipGraphExec_t* out,hipGraph_t g,hipGraphNode_t* error,char* log,size_t size) {
  using Fn=hipError_t(*)(hipGraphExec_t*,hipGraph_t,hipGraphNode_t*,char*,size_t);
  static auto next=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"hipGraphInstantiate"));
  if(!next)return hipErrorNotSupported;
  auto status=next(out,g,error,log,size);if(status==hipSuccess)inventory(g);return status;
}
extern "C" hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* out,hipGraph_t g,unsigned long long flags) {
  using Fn=hipError_t(*)(hipGraphExec_t*,hipGraph_t,unsigned long long);
  static auto next=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"hipGraphInstantiateWithFlags"));
  if(!next)return hipErrorNotSupported;
  auto status=next(out,g,flags);if(status==hipSuccess)inventory(g);return status;
}
