#!/bin/bash
# if [ -d "results" ]; then
#     rm -rf results
# fi

mkdir -p results

BITS="2 3 4 5 6 7 8"

M=1
N=4096
K=4096
# for b in $BITS; do
#     ./bin/test_any_wmma ${M} ${N} ${K} $b $b 1 > ./results/${M}x${N}x${K}_w${b}a${b}.txt
# done

./bin/test_any_cute ${M} ${N} ${K} 8 2 1 > ./results/cute_${M}x${N}x${K}_w2a8.txt
./bin/test_any_cute ${M} ${N} ${K} 2 2 1 > ./results/cute_${M}x${N}x${K}_w2a2.txt
./bin/test_any_cute ${M} ${N} ${K} 3 3 1 > ./results/cute_${M}x${N}x${K}_w3a3.txt
./bin/test_any_cute ${M} ${N} ${K} 4 4 1 > ./results/cute_${M}x${N}x${K}_w4a4.txt
./bin/test_any_cute ${M} ${N} ${K} 5 5 1 > ./results/cute_${M}x${N}x${K}_w5a5.txt
# ./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
# ./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
# ./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
# ./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt