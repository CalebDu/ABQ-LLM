#pragma once

#include "common/base.h"
#include "common/memory.h"
#include "aq_cute_atom.h"
#include <cstdint>

template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape,
    // warp shape [m, n, k] for building tiled_mma, e.g. [2, 2, 1] or [1, 2, 1]
    typename WarpShape,
    // mma op shape
    typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType, bool GridMappingXYToMN = false
    // // layout of A, B, C matrix; (MxK rowmajor) * (KxN colmajor) == (MxN rowmajor)
    // typename LayoutA = Layout::RowMajor,
    // typename LayoutB = Layout::ColumnMajor,
    // typename LayoutC = Layout::RowMajor
    >
struct AqCuteKernel {
    using type = cute::uint1_t;
    using acc_type = AccumulatorType;

    static constexpr int X_BITS = QuantType::X_BITS; // P bit
    static constexpr int W_BITS = QuantType::W_BITS; // Q bit
    // CTA tile shape
    static constexpr int BLOCK_M = ThreadBlockShape::M;
    static constexpr int BLOCK_N = ThreadBlockShape::N;
    static constexpr int BLOCK_K = ThreadBlockShape::K;

    static constexpr int MainLoop_BLOCK_M = BLOCK_M * X_BITS;
    static constexpr int MainLoop_BLOCK_N = BLOCK_N * W_BITS;
    static constexpr int MainLoop_BLOCK_K = BLOCK_K;

    static constexpr int MMA_M = MmaShape::M;
    static constexpr int MMA_N = MmaShape::N;
    static constexpr int MMA_K = MmaShape::K;

    static constexpr int Warp_M = WarpShape::M;
    static constexpr int Warp_N = WarpShape::N;
    static constexpr int Warp_K = WarpShape::K;

    static constexpr bool quant_signed = QuantType::SIGNED;
    static_assert(kThreadBlockStage > 1, "kThreadBlockStage must be greater than 1.\n");

    // shape check
    static_assert(Warp_K == 1, "only support warp_k == 1");
    static_assert(MainLoop_BLOCK_M % (MMA_N * Warp_M) == 0,
                  "BLOCK_M must be divisible by mma op m");
    static_assert(MainLoop_BLOCK_N % (MMA_N * Warp_N) == 0,
                  "BLOCK_N must be divisible by mma op n");
    static_assert(MainLoop_BLOCK_K % (MMA_K * Warp_K) == 0,
                  "BLOCK_K must be divisible by mma op k");

    // --- TiledMMA ---
    using mma_atom = typename AqCuteMmaAtom<MmaShape>::mma_atom;

    // tiled_MMA shape, [mma_m * warp_m, mma_n * warp_n, mma_k * warp_k]
    // auto compute warp_tile shape
    using TiledMMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(make_shape(Int<WarpShape::M>{}, Int<WarpShape::N>{}, Int<WarpShape::K>{}))));

    // nums of thread in thread block
    static constexpr int blockDims = size(TiledMMA{});

    // --- Smem Layout ---
    // AB Swizzle
    static constexpr int G2S_SwizzleB = 3; // 2^3 = 8
    static constexpr int G2S_SwizzleM = 7; // 2^7 = 128 uint1b_t = int128_t
    static constexpr int G2S_SwizzleS = 3; // 2^3 = 8
    // SmemLayoutAtom [8, block_k]
    using SmemABLayoutAtom =
        decltype(composition(Swizzle<G2S_SwizzleB, G2S_SwizzleM, G2S_SwizzleS>{},
                             make_layout(make_shape(Int<8>{}, Int<MainLoop_BLOCK_K>{}),
                                         make_stride(Int<MainLoop_BLOCK_K>{}, Int<1>{}))));

    // SmemALayout [x_bit * block_m, block_k, stage]
    using SmemALayout = decltype(tile_to_shape(
        SmemABLayoutAtom{},
        make_shape(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));
    // SmemBLayout [w_bit * block_n, block_k, stage]
    using SmemBLayout = decltype(tile_to_shape(
        SmemABLayoutAtom{},
        make_shape(Int<MainLoop_BLOCK_N>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));

    // !question: inputSmemSize / 8 cause illegal memory access!
    static constexpr size_t ASmemSize = cosize(SmemALayout{});
    static constexpr size_t BSmemSize = cosize(SmemBLayout{});
    static constexpr size_t inputSmemSize = ASmemSize + BSmemSize;

    // todo swizzle SmemCLayout
    // SmemCLayout [x_bit * block_m, w_bit * block_n]
    using SmemCLayout =
        decltype(make_layout(make_shape(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_N>{}),
                             make_stride(Int<MainLoop_BLOCK_N>{}, _1{})));

    static constexpr size_t outputSmem_size = cosize(SmemCLayout{}) * sizeof(acc_type);
    // total Smem usage max(A + B, C)
    static constexpr size_t SmemSize = cute::max(inputSmemSize, outputSmem_size);

    // --- Copy ---
    using Copy = AqCuteCopy<type, acc_type, ThreadBlockShape, blockDims>;
    using G2SCopyA = typename Copy::G2SCopyA;
    using G2SCopyB = typename Copy::G2SCopyB;

    using S2RCopyAtomA = typename Copy::S2RCopyAtomA;
    using S2RCopyAtomB = typename Copy::S2RCopyAtomB;
    using R2SCopyAtomC = typename Copy::R2SCopyAtomC;

    using EpilogS2RCopy = typename Copy::EpilogS2RCopy;
    using EpilogR2GCopy = typename Copy::EpilogR2GCopy;

    // --- mainloop matmul [P * block_m, K] * [Q * block_n, K]T = [P * block_m, Q * block_n] ---
    __device__ __forceinline__ void mainLoop(const int M, const int N, const int K, const int *X,
                                             const int *W, int *shared_mem_workspace);
    // --- epilogue reduce [P * block_m, Q * block_n] -> [block_m, block_n] ---
    __device__ __forceinline__ void epilogue(const int M, const int N, int *D,
                                             int *shared_mem_workspace, const half *C,
                                             bool bias = false);
};