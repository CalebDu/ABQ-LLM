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
    // warp layout [m, n, k] for building tiled_mma, e.g. [2, 2, 1] or [1, 2, 1]
    typename WarpLayout,
    // mma op shape
    typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType, bool GridMappingXYToMN = false
    // layout of A_tensor, B_tensor, C matrix; (MxK rowmajor) * (KxN colmajor) == (MxN rowmajor)
    // typename LayoutA = Layout::RowMajor,
    // typename LayoutB = Layout::ColumnMajor,
    // typename LayoutC = Layout::RowMajor
    >
struct AqCuteKernel {
    using type = cute::uint1_t;
    using acc_type = AccumulatorType;
    
    static constexpr bool GridMapping = GridMappingXYToMN;
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

    static constexpr int Warp_M = WarpLayout::M;
    static constexpr int Warp_N = WarpLayout::N;
    static constexpr int Warp_K = WarpLayout::K;

    static constexpr bool quant_signed = QuantType::SIGNED;
    static_assert(kThreadBlockStage > 1, "kThreadBlockStage must be greater than 1.\n");

    // shape check
    static_assert(Warp_K == 1, "only support warp layout k == 1");
    static_assert(MainLoop_BLOCK_M % (MMA_N * Warp_M) == 0,
                  "BLOCK_M must be divisible by (mma op m * warp layout m)");
    static_assert(MainLoop_BLOCK_N % (MMA_N * Warp_N) == 0,
                  "BLOCK_N must be divisible by (mma op n * warp layout n)");
    static_assert(MainLoop_BLOCK_K % (MMA_K * Warp_K) == 0,
                  "BLOCK_K must be divisible by (mma op k * warp layout k)");

    // --- TiledMMA ---
    using mma_atom = typename AqCuteMmaAtom<MmaShape>::mma_atom;

    // tiled_MMA shape, [mma_m * warp_m, mma_n * warp_n, mma_k * warp_k]
    // auto compute warp_tile shape
    using TiledMMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(make_shape(Int<WarpLayout::M>{}, Int<WarpLayout::N>{}, Int<WarpLayout::K>{}))));

    // nums of thread in thread block
    static constexpr int blockDims = size(TiledMMA{});

    // --- Smem Layout ---
    // AB Swizzle
    using SwizzleAtom = SwizzleAtom<MainLoop_BLOCK_M, MainLoop_BLOCK_N, MainLoop_BLOCK_K>;
    using AB_Swizzle = typename SwizzleAtom::AB_Swizzle;
    // SmemLayoutAtom [8, block_k]
    using SmemABLayoutAtom =
        decltype(composition(AB_Swizzle{},
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

    static constexpr size_t outputSmemSize = cosize(SmemCLayout{}) * sizeof(acc_type);
    // total Smem usage = max(ASmemSize + BSmemSize, CSmemSize)
    static constexpr size_t SmemSize = cute::max(inputSmemSize, outputSmemSize);

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

// Ampere, Lovelace arch or later
#if GPU_ARCH >= 80
template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape,
    // warp layout [m, n, k] for building tiled_mma, e.g. [2, 2, 1] or [1, 2, 1]
    typename WarpLayout,
    // mma op shape
    typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType, bool GridMappingXYToMN>
__device__ __forceinline__ void
AqCuteKernel<QuantType, ThreadBlockShape, WarpLayout, MmaShape, kThreadBlockStage, AccumulatorType,
             GridMappingXYToMN>::mainLoop(const int M, const int N, const int K, const int *X,
                                          const int *W, int *shared_mem_workspace)
{
    int tidx = threadIdx.x;
    int bidx_m = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int bidx_n = GridMappingXYToMN ? blockIdx.y : blockIdx.x;

    // X*WT = [P * block_m, K] * [Q * block_n, K]T = [P * block_m, Q * block_n]
    int main_loop_m = M * X_BITS;
    int main_loop_n = N * W_BITS;
    int main_loop_k = K;
    // gmem tensor
    auto A_tensor = make_tensor(make_gmem_ptr<type>(X), make_shape(main_loop_m, main_loop_k),
                                make_stride(main_loop_k, _1{}));
    auto B_tensor = make_tensor(make_gmem_ptr<type>(W), make_shape(main_loop_n, main_loop_k),
                                make_stride(main_loop_n, _1{}));

    // gmem tile tensor
    auto gA = local_tile(A_tensor, make_tile(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_K>{}),
                         make_coord(bidx_m, _)); //[P * block_m, block_k, k_loop]
    auto gB = local_tile(B_tensor, make_tile(Int<MainLoop_BLOCK_N>{}, Int<MainLoop_BLOCK_K>{}),
                         make_coord(bidx_n, _)); //[Q * block_n, block_k, k_loop]

    // smem tile tensor
    type *a_smem_ptr = reinterpret_cast<type *>(shared_mem_workspace);
    //! a_smem_ptr + (ASmemSize / 8) with alloc SmemSize/8 bytes cause illegal memory access
    type *b_smem_ptr = a_smem_ptr + ASmemSize;

    auto sA = make_tensor(make_smem_ptr<type>(a_smem_ptr),
                          SmemALayout{}); // [P * block_m, block_k, stage]
    auto sB = make_tensor(make_smem_ptr<type>(b_smem_ptr),
                          SmemBLayout{}); // [Q * block_n, block_k, stage]
    auto sC = make_tensor(make_smem_ptr<acc_type>(shared_mem_workspace),
                          SmemCLayout{}); // [P * block_m, Q * block_n]

    // tiled mma
    TiledMMA mma;
    auto thr_mma = mma.get_slice(tidx);
    auto tArA_mma = thr_mma.partition_fragment_A(gA(_, _, 0)); // [a_mma_frag_size, mma_m, mma_k]
    auto tBrB_mma = thr_mma.partition_fragment_B(gB(_, _, 0)); // [b_mma_frag_size, mma_n, mma_k]
    auto tCrC_mma = thr_mma.partition_fragment_C(sC); // [c_mma_frag_size, mma_m, mma_n]
    // set acc zero
    clear(tCrC_mma);

    // g2s load copy
    G2SCopyA a_g2s_copy;
    G2SCopyB b_g2s_copy;

    auto a_thr_g2s_copy = a_g2s_copy.get_slice(tidx);
    auto tAgA_g2s_copy = a_thr_g2s_copy.partition_S(gA); // [copy_size, copy_m, copy_k, k_loop]
    auto tAsA_g2s_copy = a_thr_g2s_copy.partition_D(sA); // [copy_size, copy_m, copy_k, stage]

    auto b_thr_g2s_copy = b_g2s_copy.get_slice(tidx);
    auto tBgB_g2s_copy = b_thr_g2s_copy.partition_S(gB); // [copy_size, copy_n, copy_k, k_loop]
    auto tBsB_g2s_copy = b_thr_g2s_copy.partition_D(sB); // [copy_size, copy_n, copy_k, stage]

    // s2r load copy
    auto a_s2r_copy = make_tiled_copy_A(S2RCopyAtomA{}, mma);
    auto a_thr_s2r_copy = a_s2r_copy.get_slice(tidx);
    auto tAsA_s2r_copy = a_thr_s2r_copy.partition_S(sA); // [copy_size, copy_m, copy_k, stage]
    auto tArA_s2r_copy = a_thr_s2r_copy.retile_D(tArA_mma); // [copy_size, copy_m, copy_k]

    auto b_s2r_copy = make_tiled_copy_B(S2RCopyAtomB{}, mma);
    auto b_thr_s2r_copy = b_s2r_copy.get_slice(tidx);
    auto tBsB_s2r_copy = b_thr_s2r_copy.partition_S(sB); // [copy_size, copy_m, copy_k, stage]
    auto tBrB_s2r_copy = b_thr_s2r_copy.retile_D(tBrB_mma); // [copy_size, copy_m, copy_k]

    // r2s store copy
    auto c_r2s_copy = make_tiled_copy_C(R2SCopyAtomC{}, mma);
    auto c_thr_r2s_copy = c_r2s_copy.get_slice(tidx);
    auto tCrC_r2s_copy = c_thr_r2s_copy.retile_S(tCrC_mma);
    auto tCsS_r2s_copy = c_thr_r2s_copy.partition_D(sC);

#if 0 // print for debug
    if (thread0()) {
        print("\nmma\n");
        print(mma);
        print("\na_g2s_copy\n");
        print(a_g2s_copy);
        print("\nb_g2s_copy\n");
        print(b_g2s_copy);
        print("\na_s2r_copy\n");
        print(a_s2r_copy);
        print("\nb_s2r_copy\n");
        print(b_s2r_copy);
        print("\nc_r2s_copy\n");
        print(c_r2s_copy);
        print("\ntArA_mma\n");
        print(tArA_mma);
        print("\ntBrB_mma\n");
        print(tBrB_mma);
        print("\ntCrC_mma\n");
        print(tCrC_mma);
        print("\ntAgA_g2s_copy\n");
        print(tAgA_g2s_copy);
        print("\ntAsA_g2s_copy\n");
        print(tAsA_g2s_copy);
        print("\ntBgB_g2s_copy\n");
        print(tBgB_g2s_copy);
        print("\ntBsB_g2s_copy\n");
        print(tBsB_g2s_copy);
        print("\ntAsA_s2r_copy\n");
        print(tAsA_s2r_copy);
        print("\ntArA_s2r_copy\n");
        print(tArA_s2r_copy);
        print("\ntBsB_s2r_copy\n");
        print(tBsB_s2r_copy);
        print("\ntBrB_s2r_copy\n");
        print(tBrB_s2r_copy);
        print("\ntCrC_r2s_copy\n");
        print(tCrC_r2s_copy);
        print("\ntCsS_r2s_copy\n");
        print(tCsS_r2s_copy);
    }
#endif
    // main loop 
}

// before Ampere
#else
template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape,
    // warp layout [m, n, k] for building tiled_mma, e.g. [2, 2, 1] or [1, 2, 1]
    typename WarpLayout,
    // mma op shape
    typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType, bool GridMappingXYToMN>
__device__ __forceinline__ void
    AqCuteKernel<QuantType, ThreadBlockShape, WarpLayout, MmaShape, kThreadBlockStage,
                 AccumulatorType, GridMappingXYToMN>::(const int M, const int N, const int K,
                                                       const int *X, const int *W,
                                                       int *shared_mem_workspace)
{
    // no implementation
}

#endif

template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape,
    // warp layout [m, n, k] for building tiled_mma, e.g. [2, 2, 1] or [1, 2, 1]
    typename WarpLayout,
    // mma op shape
    typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType, bool GridMappingXYToMN>
__device__ __forceinline__ void
AqCuteKernel<QuantType, ThreadBlockShape, WarpLayout, MmaShape, kThreadBlockStage, AccumulatorType,
             GridMappingXYToMN>::epilogue(const int M, const int N, int *D,
                                          int *shared_mem_workspace, const half *C, bool bias)
{
    int tidx = threadIdx.x;
    int bidx_m = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int bidx_n = GridMappingXYToMN ? blockIdx.y : blockIdx.x;

    auto D_tensor =
        make_tensor(make_gmem_ptr<acc_type>(D), make_shape(M, N), make_stride(N, _1{})); // [m, n]

    auto gD = local_tile(D_tensor, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                         make_coord(bidx_m, bidx_n)); //[block_m, block_n]

    auto sC = make_tensor(make_smem_ptr<type>(shared_mem_workspace),
                          SmemCLayout{}); // [P * block_m, Q * block_n]

    auto sC_tiled = local_tile(sC, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                               make_coord(_, _)); // [block_m, block_n, P, Q]

    // Epilog s2r copy
    EpilogS2RCopy c_s2r_copy;
    auto c_thr_s2r_copy = c_s2r_copy.get_slice(tidx);
    auto tCsC_s2r_copy = c_thr_s2r_copy.partition_S(sC_tiled(_, _, 0, 0));
    auto tCrC_s2r_copy = make_tensor_like(tCsC_s2r_copy);

    // Epilog r2g copy
    EpilogR2GCopy c_r2g_copy;
    auto c_thr_r2g_copy = c_r2g_copy.get_slice(tidx);
    auto tCrC_r2g_copy = c_thr_r2g_copy.retile_S(tCrC_s2r_copy);
    auto tCgC_r2g_copy = c_thr_r2g_copy.partition_D(gD);

#if 0 // print for debug
    if (thread0()) {
        print("\nc_s2r_copy\n");
        print(c_s2r_copy);
        print("\nc_r2g_copy\n");
        print(c_r2g_copy);
        print("\ntCsC_s2r_copy\n");
        print(tCsC_s2r_copy);
        print("\ntCrC_s2r_copy\n");
        print(tCrC_s2r_copy);
        print("\ntCrC_r2g_copy\n");
        print(tCrC_r2g_copy);
        print("\ntCgC_r2g_copy\n");
        print(tCgC_r2g_copy);
    }
#endif
    // epilogue 
}