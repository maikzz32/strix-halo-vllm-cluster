/* CPU-only phase timing; includes an isolated instrumented transport copy. */
#include "transport_profile.c"
static uint16_t banks[32][4][CPU_RDMA_VALUES], expected[32][CPU_RDMA_VALUES];
int main(int argc,char **argv){
 if(argc!=4)return 2;
 unsigned rank=(unsigned)atoi(argv[1]);
 void *arena=NULL;if(posix_memalign(&arena,64,CPU_RDMA_ARENA_BYTES))return 3;
 cpu_rdma_params p={.abi_version=1,.rank=rank,.world_size=4,.run_id=argv[2],.device=argv[3],.master_ipv4="192.168.100.1",.master_port=29679,.gid_index=1,.timeout_ms=5000};
 char error[512]={0};cpu_rdma_ctx *ctx=NULL;
 if(cpu_rdma_numerics_selftest(error,sizeof(error))){fprintf(stderr,"%s\n",error);return 4;}
 for(unsigned b=0;b<32;b++){
  const uint16_t *a[4];for(unsigned r=0;r<4;r++){a[r]=banks[b][r];for(unsigned i=0;i<CPU_RDMA_VALUES;i++)banks[b][r][i]=input_bits(r,b,i);}
  if(cpu_rdma_reduce_host(a,expected[b],CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16))return 5;
 }
 if(cpu_rdma_create(&p,arena,CPU_RDMA_ARENA_BYTES,&ctx,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 6;}
 uint16_t *in=(uint16_t*)((char*)arena+CPU_RDMA_INPUT_OFFSET),*out=(uint16_t*)((char*)arena+CPU_RDMA_OUTPUT_OFFSET);
 double samples[3][256][4];
 for(unsigned phase=0;phase<3;phase++){
  trace_enabled=0;

  for(unsigned it=0;it<320;it++){
   trace_enabled=((it/32)%2)==1;
   unsigned b=it%32;memcpy(in,banks[b][rank],CPU_RDMA_BYTES);
   double t=now_sec();int rc=cpu_rdma_run(ctx,in,out,CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16);double elapsed=(now_sec()-t)*1e6;
   if(rc||memcmp(out,expected[b],CPU_RDMA_BYTES)){fprintf(stderr,"run/parity error %d %s\n",rc,cpu_rdma_last_error(ctx));return 7;}
   if(it>=64){samples[phase][it-64][0]=elapsed;for(int k=0;k<3;k++)samples[phase][it-64][k+1]=trace_enabled?trace_times[k]*1e6:0.0;}
  }

 }
 printf("{\"rank\":%u,\"phases\":[",rank);
 for(unsigned phase=0;phase<3;phase++){
  printf("%s{\"alternating_32_call_blocks\":%s,\"samples_us\":[",phase?",":"","true");
  for(unsigned it=0;it<256;it++)printf("%s[%.6f,%.6f,%.6f,%.6f]",it?",":"",samples[phase][it][0],samples[phase][it][1],samples[phase][it][2],samples[phase][it][3]);
  printf("]}");
 }
 int rc=cpu_rdma_destroy(ctx);if(rc)return 8;free(arena);printf("],\"status\":\"passed\",\"verified_collectives\":960}\n");return 0;
}
