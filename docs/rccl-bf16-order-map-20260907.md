# RCCL BF16 order map, 2026-09-07

The archived four-rank RCCL reference is reproduced **bit exactly for all 327,680 values** by sixteen contiguous, fixed reduction orders. Each order follows one of the four independently logged rings. This explains the earlier failure of all 54 *global* order/tree contracts; there is no unexplained numerical corruption in these finite synthetic outputs.

This is a shape/topology-specific arithmetic hypothesis. It is not a general RCCL compatibility promise, and it has not yet been checked against independent fresh GPU inputs or the production PyNccl communicator.

## Evidence and reproducibility

The authorized reference run was `4871819146a9483684a2cdfbe50f27fa`, with four successful exits and completed teardown. Each rank's 32 x 10,240 BF16 outputs is identical. SHA256 of each 655,360-byte little-endian file: `17688c76319c552a6106d09015b5040f25e6d0aa48144340f3b034cce5590c24`.

Local command: `python tools/analyze_rccl_order_map.py results/rccl-bf16-reference-20260907`. Results: `order-map.json` and `order-map-arrays.npz` in that directory. The analyzer compares 48 permutation contracts plus six balanced-tree contracts and separately reconstructs the fixed source/log-derived profile. The latter has **zero mismatches**. No GPU, network test, or runtime modification was needed for this analysis.

All 10,240 indices have at least one contract exact across all 32 banks. Greedy common-contract ranges sometimes extend a boundary by two positions: the generator's `index % 16 == 0` and `== 1` cancellation cases are constant across banks and leave those particular positions ambiguous. The fixed 768/256-element grid matches without those data-dependent boundary extensions. Swapping the first two finite operands is also numerically indistinguishable; ring logs resolve the intended topology.

## Independent topology evidence

All three prior `rccl-tp4-20k-auto4`, `auto4-repeat`, and `auto4-final` result directories dated `20260906` contain the same rank0/node15 log evidence: lines 46–49 list the four rings below, line 192 reports four collective channels, and line 203 selects Ring/LL for 20,480 bytes over channels 0–3. Their earlier `ringGraph nChannels 6` is a topology-search intermediate, not six active collective channels. The newer reference collector did not enable NCCL INFO, so its own logs do not independently repeat these ring lines.

| Channel | BF16 index range | Ring | Four chunk lengths | Accumulation orders |
|---|---|---|---|---|
| 0 | [0,3072) | 0,2,1,3 | 768 each | 2130, 1302, 3021, 0213 |
| 1 | [3072,6144) | 0,2,3,1 | 768 each | 2310, 3102, 1023, 0231 |
| 2 | [6144,9216) | 0,3,1,2 | 768 each | 3120, 1203, 2031, 0312 |
| 3 | [9216,10240) | 0,1,3,2 | 256 each | 1320, 3201, 2013, 0132 |

For ring array `r` and chunk number `j`, the reduction order is `r[j+1:] + r[:j+1]`: the owner's successor contributes first and the owner contributes last. Every addition is rounded to BF16 with round-to-nearest-even.

## Source-derived partition, independent of output fitting

Public RCCL `develop` schedules all-reduce with traffic factor 2, multiplied by 4 for LL. A 16-KiB minimum traffic cell gives `cellBytes = alignUp(ceil(16384 / 8),16) = 2048`. The 20,480-byte input therefore has ten cells. For one operation over four available channels, modeled traffic per channel is `20480*8/4 = 40960` bytes, so each ordinary channel receives `ceil(40960/16384) = 3` cells. The continuous-byte-distribution fields become 3072/3072/1024 BF16 values (lo/mid/hi), with two middle channels: **3072+3072+3072+1024**. These traffic factors are scheduler weights, not measured wire byte counts. See [RCCL scheduler](https://github.com/ROCm/rccl/blob/develop/src/enqueue.cc#L628-L756) and [traffic-factor definition](https://github.com/ROCm/rccl/blob/develop/src/enqueue.cc#L143-L150).

`ncclCollCbdPart` converts those fields into per-channel offsets/counts. The Ring final loop splits a channel into four chunks and aligns to `16/sizeof(BF16)=8` elements: `alignUp(ceil(3072/4),8)=768`, or 256 for the final channel. The send/reduce schedule ends reduction at chunk owner `ringIx`, followed by copies. See [partition decoder](https://github.com/ROCm/rccl/blob/develop/src/include/device.h#L399-L423) and [Ring schedule](https://github.com/ROCm/rccl/blob/develop/src/device/all_reduce.h#L74-L163).

The public AMD default LL buffer is 8 lines/thread x 256 threads x 8 steps x 16 bytes = 262,144 bytes, yielding a 16,384-byte LL data chunk after division by eight steps and two for LL packing. This is larger than the 1,536-byte maximum final chunk here, consistent with a single final loop. See [AMD default buffer setup](https://github.com/ROCm/rccl/blob/develop/src/rccl_wrap.cc#L648-L688), [thread constant](https://github.com/ROCm/rccl/blob/develop/src/include/rccl_common.h#L47), and [LL chunking](https://github.com/ROCm/rccl/blob/develop/src/enqueue.cc#L2288-L2295). This derivation assumes no LL-buffer override.

The installed library was independently identified as RCCL 2.30.4, ROCm 10 development build `6b0e43f3`. Its exact installed source revision has **not** been matched to public sources. The public source is a corroborating mechanism, not proof of binary identity. Likewise, actual production grouping, ring assignment, channel counts, pointer alignment, and a changed library can alter this contract.

## Frozen profile and next gate

`tools/cpu_rdma_ring4_reference.py` exports `reduce_rccl_ring4_bits(inputs)` for native NumPy uint16 shape `(4,10240)` and returns `(10240,)`. It reuses the established encode/decode implementation. Profile name: `rccl-2.30.4-ring-ll-4ch-20k-0213-0231-0312-0132-v1`; canonical profile SHA256: `2acfe604f80dbece21e2ee77e9accbc5952bf356833a1d5e2b722513a2c23f80`.

Four local tests pass, including all archived bytes and independently generated finite CPU inputs against the scalar BF16 oracle. The archive test skips explicitly when the external raw evidence is absent. NaN/Inf conversion and gradual underflow remain available; matching GPU compiler choices for exceptional NaN signs is not established by finite reference banks.

Before a model experiment, an isolated optional C mode must match these archived bits and the existing scalar exceptional-value semantics. A separate startup calibration must compare **independent fresh banks** against the **actual communicator**, both eager and graph, with repetitions and relevant input offsets. Calibration failure must keep the candidate inactive. No serving integration, model-output parity, quality claim, or TPS claim follows from this report.
