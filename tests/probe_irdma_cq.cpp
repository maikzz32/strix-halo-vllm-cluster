#include <infiniband/verbs.h>
#include <initializer_list>
#include <cerrno>
#include <cstdio>
#include <cstring>
int main() {
 int n=0;auto list=ibv_get_device_list(&n);ibv_context* c=nullptr;
 for(int i=0;i<n;i++)if(!strcmp(ibv_get_device_name(list[i]),"rocep197s0f1"))c=ibv_open_device(list[i]);
 if(!c)return 2;
 ibv_device_attr d{};if(ibv_query_device(c,&d))return 3;
 printf("max_cqe=%d\n",d.max_cqe);
 printf("max_srq=%d max_srq_wr=%d max_srq_sge=%d\n",d.max_srq,d.max_srq_wr,d.max_srq_sge);
 const unsigned long full=IBV_WC_EX_WITH_BYTE_LEN|IBV_WC_EX_WITH_IMM|IBV_WC_EX_WITH_QP_NUM|IBV_WC_EX_WITH_SRC_QP|IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
 for(int size: {1,16384})for(int mode=0;mode<4;mode++){
  ibv_cq_init_attr_ex a{};a.cqe=size;
  if(mode){a.wc_flags=full;a.comp_mask=IBV_CQ_INIT_ATTR_MASK_FLAGS;a.flags=IBV_CREATE_CQ_ATTR_SINGLE_THREADED;}
  if(mode==2)a.wc_flags&=~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
  if(mode==3){a.wc_flags&=~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;a.wc_flags&=~IBV_WC_EX_WITH_SRC_QP;}
  errno=0;auto q=ibv_create_cq_ex(c,&a);int e=errno;
  printf("size=%d mode=%d success=%d errno=%d message=%s\n",size,mode,q!=nullptr,e,strerror(e));
  if(q)ibv_destroy_cq(ibv_cq_ex_to_cq(q));
 }
 ibv_close_device(c);ibv_free_device_list(list);return 0;
}
