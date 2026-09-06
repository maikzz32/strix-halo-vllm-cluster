/* Include after scalar reduce_four(). No fast-math compilation is permitted.
 * Fixed rank order and BF16 RNE match that scalar reference. The ordinary
 * entry point checks AVX2 before entering target-specific code.
 */
#ifndef CPU_RDMA_REDUCE_AVX2_H
#define CPU_RDMA_REDUCE_AVX2_H

#include <stddef.h>
#include <stdint.h>

static void reduction_scalar_range(const uint16_t *a[4], uint16_t *out,
                                   size_t begin, size_t end, int round_each) {
    for (size_t i = begin; i < end; ++i) {
        uint16_t values[4] = {a[0][i], a[1][i], a[2][i], a[3][i]};
        out[i] = reduce_four(values, round_each);
    }
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>

static int reduction_avx2_available(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
}

/* Return eight BF16 encodings in the low 16 bits of eight uint32 lanes.
 * The special branch matches scalar to_bf16 for infinities and canonical
 * signed quiet NaNs, including finite additions that overflow to infinity.
 */
__attribute__((target("avx2"), always_inline))
static inline __m256i reduction_bf16_rne8(__m256 value) {
    const __m256i bits = _mm256_castps_si256(value);
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    const __m256i fraction = _mm256_set1_epi32(0x007fffff);
    const __m256i high = _mm256_srli_epi32(bits, 16);
    const __m256i tie = _mm256_and_si256(high, _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_srli_epi32(
        _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), tie)), 16);
    const __m256i special_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(bits, exponent), exponent);
    const __m256i infinity_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(bits, fraction), _mm256_setzero_si256());
    const __m256i canonical_nan = _mm256_or_si256(
        _mm256_and_si256(high, _mm256_set1_epi32(0x8000)),
        _mm256_set1_epi32(0x7fc0));
    const __m256i special = _mm256_blendv_epi8(canonical_nan, high, infinity_mask);
    return _mm256_blendv_epi8(rounded, special, special_mask);
}

__attribute__((target("avx2")))
static void reduction_avx2_impl(const uint16_t *a[4], uint16_t *out,
                                size_t count, int round_each) {
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    size_t i = 0;
    const size_t vector_count = count & ~(size_t)7;
    for (; i < vector_count; i += 8) {
        __m256i words[4];
        __m256i nonfinite = _mm256_setzero_si256();
        for (int rank = 0; rank < 4; ++rank) {
            const __m128i packed = _mm_loadu_si128((const __m128i *)(a[rank] + i));
            words[rank] = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
            nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(
                _mm256_and_si256(words[rank], exponent), exponent));
        }
        /* NaN payload/sign selection can depend on scalar operand ordering.
         * Preserve it by using the original reference for the entire block.
         */
        if (_mm256_movemask_epi8(nonfinite)) {
            reduction_scalar_range(a, out, i, i + 8, round_each);
            continue;
        }
        __m256 accumulator = _mm256_castsi256_ps(words[0]);
        for (int rank = 1; rank < 4; ++rank) {
            accumulator = _mm256_add_ps(accumulator, _mm256_castsi256_ps(words[rank]));
            if (round_each) {
                accumulator = _mm256_castsi256_ps(
                    _mm256_slli_epi32(reduction_bf16_rne8(accumulator), 16));
            }
        }
        const __m256i result = reduction_bf16_rne8(accumulator);
        const __m128i packed = _mm_packus_epi32(
            _mm256_castsi256_si128(result), _mm256_extracti128_si256(result, 1));
        _mm_storeu_si128((__m128i *)(out + i), packed);
    }
    reduction_scalar_range(a, out, i, count, round_each);
}

static void reduce_avx2(const uint16_t *a[4], uint16_t *out,
                        size_t count, int round_each) {
    if (reduction_avx2_available())
        reduction_avx2_impl(a, out, count, round_each);
    else
        reduction_scalar_range(a, out, 0, count, round_each);
}

#else

static int reduction_avx2_available(void) { return 0; }

static void reduce_avx2(const uint16_t *a[4], uint16_t *out,
                        size_t count, int round_each) {
    reduction_scalar_range(a, out, 0, count, round_each);
}

#endif
#endif
