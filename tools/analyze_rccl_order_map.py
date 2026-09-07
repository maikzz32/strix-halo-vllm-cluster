#!/usr/bin/env python3
"""Pure-local per-index mapping of archived RCCL bits to54 rounding contracts."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_rccl_bf16_reference import decode, encode
from cpu_rdma_reference import COUNT, PERIOD, input_bits
from cpu_rdma_ring4_reference import PROFILE, PROFILE_SHA256, partition_orders, reduce_rccl_ring4_bits


def candidates(values):
    for each in (False, True):
        label = 'bf16_each_add' if each else 'fp32_then_bf16'
        for order in itertools.permutations(range(4)):
            total = values[order[0]].copy()
            for rank in order[1:]:
                total = total + values[rank]
                if each:
                    total = decode(encode(total))
            yield label + ':' + ''.join(map(str, order)), encode(total)
        for order in ((0,1,2,3), (0,2,1,3), (0,3,1,2)):
            a,b,c,d = order
            left, right = values[a] + values[b], values[c] + values[d]
            if each:
                left, right = decode(encode(left)), decode(encode(right))
            yield label + ':balanced:' + ''.join(map(str,order)), encode(left + right)


def labels_for(mask, labels):
    return [name for i,name in enumerate(labels) if int(mask) & (1 << i)]


def common_segments(masks, labels):
    """Greedy maximal spans sharing at least one exact32-bank contract.

    This reports data-supported intervals, not an inferred kernel partition.
    Ambiguous constant/cancellation positions may extend a boundary.
    """
    result=[]; start=0; common=int(masks[0])
    for index in range(1,len(masks)):
        value=int(masks[index]); overlap=common & value
        if overlap or (common==0 and value==0):
            common=overlap
        else:
            result.append({'start':start,'end_exclusive':index,'length':index-start,
                           'exact_contracts':labels_for(common,labels)})
            start=index;common=value
    result.append({'start':start,'end_exclusive':len(masks),'length':len(masks)-start,
                   'exact_contracts':labels_for(common,labels)})
    return result


def analyze(directory):
    paths=sorted(directory.glob('rank*-node*.bf16.bin'))
    if len(paths)!=4:
        raise ValueError('four archived output files required')
    outputs=[np.fromfile(path,dtype='<u2') for path in paths]
    if any(a.size!=COUNT*PERIOD for a in outputs) or not all(np.array_equal(outputs[0],a) for a in outputs[1:]):
        raise ValueError('wrong shape or unequal rank outputs')
    actual=outputs[0].reshape(PERIOD,COUNT)
    inputs=np.array([[[input_bits(rank,bank,index) for index in range(COUNT)]
                       for bank in range(PERIOD)] for rank in range(4)],dtype=np.uint16)
    values=decode(inputs)
    labels=[]; predictions=[]; scores=[]; masks=np.zeros(COUNT,dtype=np.uint64)
    any_match=np.zeros_like(actual,dtype=bool)
    for number,(label,predicted) in enumerate(candidates(values)):
        labels.append(label); predictions.append(predicted)
        equal=predicted==actual; counts=equal.sum(axis=0)
        scores.append(counts)
        masks |= ((counts==PERIOD).astype(np.uint64) << np.uint64(number))
        any_match |= equal
    scores=np.array(scores,dtype=np.uint8);predictions=np.array(predictions,dtype=np.uint16)
    best_matches=scores.max(axis=0)
    unknown=np.flatnonzero(masks==0)
    count_hist={str(i):int(np.count_nonzero(best_matches==i)) for i in range(PERIOD+1) if np.any(best_matches==i)}
    mask_values,mask_counts=np.unique(masks,return_counts=True)
    masks_table=[{'mask_hex':f'{int(m):014x}','index_count':int(n),'contracts':labels_for(m,labels)}
                 for m,n in zip(mask_values,mask_counts)]
    unexplainable=np.argwhere(~any_match)
    examples=[]
    for bank,index in unexplainable[:24]:
        bank,index=int(bank),int(index)
        bits=sorted(set(int(x) for x in predictions[:,bank,index]))
        examples.append({'bank':bank,'index':index,'input_bits':inputs[:,bank,index].tolist(),
                         'input_values':values[:,bank,index].tolist(),'actual_bits':int(actual[bank,index]),
                         'actual_value':float(decode(actual[bank,index])),
                         'possible_bits':bits,'possible_values':decode(bits).tolist()})
    block_tests=[]
    for size in (8,16,32,64,128,256,512,640,768,1024,1280,2048,2560,4096,5120):
        blocks=[]
        for start in range(0,COUNT,size):
            end=min(start+size,COUNT)
            common=int(np.bitwise_and.reduce(masks[start:end]))
            blocks.append({'start':start,'end_exclusive':end,'contracts':labels_for(common,labels)})
        block_tests.append({'block_values':size,'blocks':len(blocks),
                            'exact_blocks':sum(bool(b['contracts']) for b in blocks),
                            'all_blocks_exact':all(bool(b['contracts']) for b in blocks),
                            'mapping':blocks if size>=128 else None})
    segments=common_segments(masks,labels)
    reconstructed=np.array([reduce_rccl_ring4_bits(inputs[:,bank,:]) for bank in range(PERIOD)])
    fixed_chunks=[{'start':start,'end_exclusive':end,'order':list(order),
                   'mismatches':int(np.count_nonzero(reconstructed[:,start:end]!=actual[:,start:end]))}
                  for start,end,order in partition_orders()]
    report={'shape':[PERIOD,COUNT],'candidate_count':len(labels),'labels':labels,
        'all_rank_outputs_equal':True,'indices_with_one_contract_exact_across_all_banks':int(np.count_nonzero(masks)),
        'indices_without_fixed_exact_contract':int(unknown.size),'first_unknown_indices':unknown[:64].tolist(),
        'best_bank_matches_histogram':count_hist,'elements_outside_every_contract':int(unexplainable.shape[0]),
        'outside_every_contract_examples':examples,'mask_classes':masks_table,
        'common_segments':segments,'block_tests':block_tests,
        'source_and_log_derived_profile':PROFILE,'profile_sha256':PROFILE_SHA256,
        'profile_mismatches':int(np.count_nonzero(reconstructed!=actual)),
        'profile_chunks':fixed_chunks,
        'reconstructed_output_sha256':hashlib.sha256(reconstructed.astype('<u2').tobytes()).hexdigest(),
        'global_contract_scores':[{'contract':name,'mismatches':int(PERIOD*COUNT-scores[i].sum()),
                                  'indices_exact_all_banks':int(np.count_nonzero(scores[i]==PERIOD))}
                                 for i,name in enumerate(labels)],
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'note':'Exactness on fixed32 synthetic banks; order labels may be numerically indistinguishable. No runtime change or performance/quality claim.'}
    np.savez_compressed(directory/'order-map-arrays.npz',exact_contract_mask=masks,bank_match_counts=scores,
                        actual_bits=actual,labels=np.array(labels))
    (directory/'order-map.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();r=analyze(args.directory)
    print(json.dumps({k:r[k] for k in ('candidate_count','indices_with_one_contract_exact_across_all_banks',
        'indices_without_fixed_exact_contract','best_bank_matches_histogram','elements_outside_every_contract',
        'first_unknown_indices','outside_every_contract_examples')},indent=2))
    print('common_segments',len(r['common_segments']))
    if len(r['common_segments'])<40: print(json.dumps(r['common_segments'],indent=2))


if __name__=='__main__':main()
