#!/bin/bash

mkdir -p results

BITS="2 3 4 5 6 7 8"

M=1
N=4096
K=4096


# ./bin/test_any_cute ${M} ${N} ${K} 2 2 1 > ./results/${M}x${N}x${K}_w2a2.txt

ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum --csv ./bin/test_any_cute ${M} ${N} ${K} 2 2 1 1 0 > ./results/cute_bankconflict_${M}x${N}x${K}_w2a2.txt