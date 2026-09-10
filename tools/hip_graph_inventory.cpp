// Read-only graph topology capture at instantiation. No replay interception.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <link.h>
#include <cstring>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <unordered_map>

static void* resolve(const char* symbol) {
  if(void* p=dlsym(RTLD_NEXT,symbol)) return p;
  void* handle=nullptr;
  dl_iterate_phdr([](dl_phdr_info* info,size_t,void* data) {
    if(strstr(info->dlpi_name,"libamdhip64.so")) {
      *static_cast<void**>(data)=dlopen(info->dlpi_name,RTLD_LAZY|RTLD_NOLOAD);
      return *static_cast<void**>(data)?1:0;
    }
    return 0;
  },&handle);
  if(!handle)return nullptr;
  void* p=dlsym(handle,symbol);dlclose(handle);return p;
}
#define QUERY(name, ...) ([&]() { auto fn=reinterpret_cast<decltype(&name)>(resolve(#name)); return fn?fn(__VA_ARGS__):hipErrorNotSupported; }())
static void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error("HIP graph query failed"); }
static void graph_json(FILE* f, hipGraph_t g, unsigned depth) {
  if(depth>8) throw std::runtime_error("nested graph depth exceeded");
  size_t count=0; check(QUERY(hipGraphGetNodes,g,nullptr,&count));
  if(count>100000) throw std::runtime_error("node bound exceeded");
  std::vector<hipGraphNode_t> nodes(count);
  check(QUERY(hipGraphGetNodes,g,nodes.data(),&count));
  std::unordered_map<hipGraphNode_t,size_t> indices;
  for(size_t i=0;i<count;++i) indices[nodes[i]]=i;
  fprintf(f,"{\"node_count\":%zu,\"nodes\":[",count);
  for(size_t i=0;i<count;++i) {
    hipGraphNodeType type;check(QUERY(hipGraphNodeGetType,nodes[i],&type));
    size_t n=0;check(QUERY(hipGraphNodeGetDependencies,nodes[i],nullptr,&n));
    std::vector<hipGraphNode_t> deps(n);
    if(n)check(QUERY(hipGraphNodeGetDependencies,nodes[i],deps.data(),&n));
    fprintf(f,"%s{\"index\":%zu,\"type\":%d,\"dependencies\":[",i?",":"",i,int(type));
    for(size_t j=0;j<n;++j)fprintf(f,"%s%zu",j?",":"",indices.at(deps[j]));
    fprintf(f,"]");
    if(type==hipGraphNodeTypeMemcpy) {
      hipMemcpy3DParms p{};check(QUERY(hipGraphMemcpyNodeGetParams,nodes[i],&p));
      fprintf(f,",\"copy\":{\"kind\":%d,\"width\":%zu,\"height\":%zu,\"depth\":%zu}",int(p.kind),p.extent.width,p.extent.height,p.extent.depth);
    }
    if(type==hipGraphNodeTypeGraph) {
      hipGraph_t child;check(QUERY(hipGraphChildGraphNodeGetGraph,nodes[i],&child));
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
  static auto next=reinterpret_cast<Fn>(resolve("hipGraphInstantiate"));
  if(!next)return hipErrorNotSupported;
  auto status=next(out,g,error,log,size);if(status==hipSuccess)inventory(g);return status;
}
extern "C" hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* out,hipGraph_t g,unsigned long long flags) {
  using Fn=hipError_t(*)(hipGraphExec_t*,hipGraph_t,unsigned long long);
  static auto next=reinterpret_cast<Fn>(resolve("hipGraphInstantiateWithFlags"));
  if(!next)return hipErrorNotSupported;
  auto status=next(out,g,flags);if(status==hipSuccess)inventory(g);return status;
}
