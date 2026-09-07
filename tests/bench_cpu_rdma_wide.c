/* CPU-only exactness and cost study; never creates a verbs or HIP context. */
#include "cpu_rdma_transport.c"
#include "cpu_rdma_reduce_avx512.h"
#include "wide_entry.h"
static uint16_t inputs[4][65568],ref[65568],wide[65568];
int main(int argc,char**argv){
 if(argc!=4||!reduction_avx512_available())return 2;
 char error[512];if(cpu_rdma_numerics_selftest(error,sizeof(error)))return 3;
 unsigned control=_mm_getcsr()&~0x3fu,checks=0;
 const uint16_t *a[4]={inputs[0],inputs[1],inputs[2],inputs[3]};
 for(unsigned pattern=0;pattern<9;pattern++){
  for(unsigned i=0;i<65568;i++)for(unsigned r=0;r<4;r++){
   uint16_t v;
   if(pattern<4)v=r==pattern?(uint16_t)i:(r&1)?0xbf80:0x3f80;
   else if(pattern==4)v=(uint16_t)(i+r*16381u);
   else if(pattern==5)v=(uint16_t)i;
   else if(pattern==6)v=(r==0||r==2)?(uint16_t)(i^(r==2?0x8000:0)):0x3f80;
   else if(pattern==7)v=(uint16_t)((i*40503u+r*49157u)^((i+r)>>3));
   else v=input_bits(r,i%32,i);
   inputs[r][i]=v;
  }
  for(unsigned mode=0;mode<2;mode++){
   reduce_avx2(a,ref,65568,mode);reduce_avx512_impl(a,wide,65568,mode);
   if(memcmp(ref,wide,sizeof(ref))){fprintf(stderr,"parity pattern%u mode%u\n",pattern,mode);return 4;}checks++;
   for(unsigned offset=0;offset<3;offset++)for(unsigned count=0;count<66;count++){
    const uint16_t *u[4];for(int r=0;r<4;r++)u[r]=a[r]+offset;
    memset(ref,0xa5,sizeof(ref));memset(wide,0xa5,sizeof(wide));
    reduce_avx2(u,ref+1,count,mode);reduce_avx512_impl(u,wide+1,count,mode);
    if(memcmp(ref,wide,sizeof(ref)))return 5;checks++;
   }
  }
 }
 if((_mm_getcsr()&~0x3fu)!=control)return 6;
 enum{BANKS=32,LOOPS=2048,SAMPLES=5};
 uint16_t *data=aligned_alloc(64,BANKS*5*CPU_RDMA_BYTES);if(!data)return 7;
 for(unsigned b=0;b<BANKS;b++){
  const uint16_t *u[4];for(unsigned r=0;r<4;r++){
   u[r]=data+(b*5+r)*CPU_RDMA_VALUES;
   for(unsigned i=0;i<CPU_RDMA_VALUES;i++)data[(b*5+r)*CPU_RDMA_VALUES+i]=input_bits(r,b,i);
  }
  if(cpu_rdma_reduce_host(u,ref,CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16)||cpu_rdma_reduce_host_wide(u,wide,CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16)||memcmp(ref,wide,CPU_RDMA_BYTES))return 8;checks++;
 }
 double samples[2][2][SAMPLES];
 for(unsigned group=0;group<2;group++)for(unsigned sample=0;sample<SAMPLES;sample++)for(unsigned order=0;order<2;order++){
  unsigned method=order^(sample&1),banks=group?BANKS:1;double t=now_sec();
  for(unsigned it=0;it<LOOPS;it++){
   unsigned b=it%banks;const uint16_t *u[4];for(unsigned r=0;r<4;r++)u[r]=data+(b*5+r)*CPU_RDMA_VALUES;
   int rc=method?cpu_rdma_reduce_host_wide(u,data+(b*5+4)*CPU_RDMA_VALUES,CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16):cpu_rdma_reduce_host(u,data+(b*5+4)*CPU_RDMA_VALUES,CPU_RDMA_VALUES,CPU_RDMA_RCCL_RING4_BF16);
   if(rc)return 9;
  }
  samples[group][method][sample]=(now_sec()-t)*1e6/LOOPS;
 }
 printf("{\"rank\":%s,\"status\":\"passed\",\"checks\":%u,\"samples\":[",argv[1],checks);
 for(unsigned group=0;group<2;group++)for(unsigned method=0;method<2;method++){
  printf("%s{\"banks\":%u,\"method\":\"%s\",\"us\":[",group||method?",":"",group?BANKS:1,method?"avx512":"avx2");
  for(unsigned s=0;s<SAMPLES;s++)printf("%s%.9f",s?",":"",samples[group][method][s]);printf("]}");
 }
 printf("]}\n");free(data);return 0;
}
