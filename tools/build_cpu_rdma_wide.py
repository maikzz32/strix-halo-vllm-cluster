"""Generate an isolated transport using exact AVX-512 with AVX2 fallback."""
import hashlib
BASE_SHA='b44cc2ff7f1f6dfb72126499a985057446a586f5b4df6afd27e4619248f70fde'
def generate(source):
 if hashlib.sha256(source.encode()).hexdigest()!=BASE_SHA:raise ValueError('Unexpected transport source')
 anchor='#include "cpu_rdma_reduce_avx2.h"'
 if source.count(anchor)!=1 or source.count('reduce_avx2(')!=3:raise ValueError('Transport dispatch changed')
 source=source.replace('reduce_avx2(', 'reduce_wide_checked(')
 addition="""
#include "cpu_rdma_reduce_avx512.h"
static void reduce_wide_checked(const uint16_t *a[4],uint16_t *out,size_t count,int mode){
    if(reduction_avx512_available())reduce_avx512_impl(a,out,count,mode);
    else reduce_avx2(a,out,count,mode);
}
"""
 return source.replace(anchor,anchor+addition,1)
