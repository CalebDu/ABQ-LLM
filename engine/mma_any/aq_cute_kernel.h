#pragma once

#include "common/base.h"
#include "common/memory.h"
#include "aq_cute_atom.h"
#include "cute/numeric/int.hpp"
#include <cstdint>

template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape,
    // warp shape [m, n, k] for tiled_mma, e.g. [2, 2, 1] or [1, 2, 1]
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

    static constexpr int X_BITS = QuantType::X_BITS;
    static constexpr int W_BITS = QuantType::W_BITS;
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

    static constexpr bool GridMapping = GridMappingXYToMN;

    using mma_atom = typename AqCuteMmaAtom<MmaShape>::mma_atom;

    // tiled CTA tile shape, [mma_m * warp_m, mma_n * warp_n, mma_k * warp_k]
    // auto compute warp_tile shape
    using mma = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(make_shape(Int<WarpShape::M>{}, Int<WarpShape::N>{}, Int<WarpShape::K>{}))));

    static constexpr int kThread = size(mma{});

    using SmemLayout = AqCuteSmemLayout<QuantType, ThreadBlockShape, kThreadBlockStage>;

    using SmemALayout = typename SmemLayout::SmemALayout;
    using SmemBLayout = typename SmemLayout::SmemBLayout;

    // mainloop interface
    __device__ __forceinline__ void mainLoop(const int M, const int N, const int K, const int *X,
                                             const int *W, int *shared_mem_workspace);

    __device__ __forceinline__ void epilogue(const int M, const int N, int *D,
                                             int *shared_mem_workspace, const half *C,
                                             bool bias = false);
};