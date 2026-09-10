// Probe the documented plugin ABI in a separate bounded process.
#include "nccl_net.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  setvbuf(stdout,nullptr,_IONBF,0);
  void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());return 3;}
  auto* p=(ncclNet_v8_t*)dlsym(lib,"ncclNetPlugin_v8");
  if(!p){fprintf(stderr,"dlsym: %s\n",dlerror());return 4;}
  printf("PLUGIN_NAME=%s\n",p->name);
  printf("INIT_BEGIN\n");
  auto rc=p->init(nullptr);printf("INIT_RESULT=%d\n",int(rc));
  if(rc!=ncclSuccess)return 5;
  int count=0;rc=p->devices(&count);printf("DEVICES_RESULT=%d COUNT=%d\n",int(rc),count);
  if(rc!=ncclSuccess||count!=1)return 6;
  ncclNetProperties_v8_t props{};
  printf("PROPERTIES_BEGIN\n");rc=p->getProperties(0,&props);
  printf("PROPERTIES_RESULT=%d\n",int(rc));
  if(rc!=ncclSuccess)return 7;
  printf("DEVICE=%s PTR_SUPPORT=%d SPEED_MBPS=%d\n",props.name,props.ptrSupport,props.speed);
  printf("PROBE_COMPLETE\n");
  return 0;
}
