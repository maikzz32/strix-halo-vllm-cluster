/* Experimental exact AVX-512 reducer; include after the existing scalar/AVX2 helpers.
 * Integer BF16 RNE preserves subnormals and canonical signed NaNs. No fast math. */
#ifndef STRIX_REDUCE_AVX512_H
#define STRIX_REDUCE_AVX512_H
static int reduction_avx512_available(void){
 __builtin_cpu_init();return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw");
}
__attribute__((target("avx512f,avx512bw"),always_inline))
static inline __m512i reduction_bf16_rne16(__m512 value){
 __m512i bits=_mm512_castps_si512(value),high=_mm512_srli_epi32(bits,16);
 __m512i rounded=_mm512_srli_epi32(_mm512_add_epi32(bits,_mm512_add_epi32(_mm512_set1_epi32(0x7fff),_mm512_and_si512(high,_mm512_set1_epi32(1)))),16);
 __mmask16 special=_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits,_mm512_set1_epi32(0x7f800000)),_mm512_set1_epi32(0x7f800000));
 __mmask16 infinity=_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits,_mm512_set1_epi32(0x007fffff)),_mm512_setzero_si512());
 __m512i nan=_mm512_or_si512(_mm512_and_si512(high,_mm512_set1_epi32(0x8000)),_mm512_set1_epi32(0x7fc0));
 return _mm512_mask_mov_epi32(rounded,special,_mm512_mask_mov_epi32(nan,infinity,high));
}
__attribute__((target("avx512f,avx512bw")))
static void reduce_avx512_impl(const uint16_t *a[4],uint16_t *out,size_t count,int round_each){
 size_t i=0;const size_t end=count&~(size_t)15;
 for(;i<end;i+=16){
  __m512i words[4];__mmask16 nonfinite=0;
  for(int r=0;r<4;r++){
   words[r]=_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(a[r]+i))),16);
   nonfinite|=_mm512_cmpeq_epi32_mask(_mm512_and_si512(words[r],_mm512_set1_epi32(0x7f800000)),_mm512_set1_epi32(0x7f800000));
  }
  if(nonfinite){reduction_scalar_range(a,out,i,i+16,round_each);continue;}
  __m512 acc=_mm512_castsi512_ps(words[0]);__m512i result=_mm512_setzero_si512();
  for(int r=1;r<4;r++){
   acc=_mm512_add_ps(acc,_mm512_castsi512_ps(words[r]));
   if(round_each){result=reduction_bf16_rne16(acc);acc=_mm512_castsi512_ps(_mm512_slli_epi32(result,16));}
  }
  if(!round_each)result=reduction_bf16_rne16(acc);
  _mm256_storeu_si256((__m256i*)(out+i),_mm512_cvtepi32_epi16(result));
 }
 reduction_scalar_range(a,out,i,count,round_each);
}
#endif
