#pragma once

#include "common/base.h"
#include "common/memory.h"
#include "aq_cute_atom.h"
#include <cstdint>
#define tid 0 // thread for debug
template <class... CopyArgs, class PredTensor, class SrcEngine, class SrcLayout, class DstEngine,
          class DstLayout, class StripTuple, class ZfillTuple>
__device__ __forceinline__ static void
copy_strip_zfill(Copy_Atom<CopyArgs...> const &copy, PredTensor const &pred,
                 Tensor<SrcEngine, SrcLayout> const &src, Tensor<DstEngine, DstLayout> dst,
                 StripTuple const &strip_bound, ZfillTuple const &zfill_bound)
{
    static_assert(SrcLayout::rank == DstLayout::rank, "dst and src mismatch rank ");
    constexpr int Rank = SrcLayout::rank;
    // print_type(Rank);
    auto src_v = group_modes<1, Rank>(src); // [copy, copy_m * copy_n]
    auto dst_v = group_modes<1, Rank>(dst); // [copy, copy_m * copy_n]
    auto pred_v = group_modes<1, Rank>(pred); // [copy, copy_m * copy_n]
#pragma unroll
    for (int idx = 0; idx < size<1>(pred_v); idx++) {
        auto pred_coord = pred_v(_0{}, idx);
        // if (thread(tid)) {
        //     print("\npred\n");
        //     print(pred_coord);
        //     print("\nstrip_bound\n");
        //     print(strip_bound);
        //     print("\nstrip check\n");
        //     print(elem_less(pred_coord, strip_bound));
        // }
        // strip data OOB block tile
        if (elem_less(pred_coord, strip_bound)) {
            // fill zeros OOB global shape into block tile
            copy_if(
                copy,
                [&](auto... coords) { return elem_less(pred_v(_0{}, coords...), zfill_bound); },
                src_v(_, _), dst_v(_, _));
        }
    }
}

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

    static constexpr int kStage = kThreadBlockStage;
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

    static_assert(BLOCK_M % 4 == 0, "BLOCK_M must be divisible by 4");
    static_assert(BLOCK_N % 8 == 0, "BLOCK_N must be divisible by 8");
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
    using SwizzleAtom = SwizzleAtom<BLOCK_M, BLOCK_N, BLOCK_K>;
    using AB_Swizzle = typename SwizzleAtom::AB_Swizzle;
    // SmemLayoutAtom [8, block_k]
    using SmemABLayoutAtom = decltype(composition(
        AB_Swizzle{}, make_layout(make_shape(Int<8>{}, Int<MainLoop_BLOCK_K>{}),
                                  make_stride(Int<MainLoop_BLOCK_K>{}, Int<1>{}))));

    // SmemALayout [x_bit * block_m, block_k, stage]
    using SmemALayout = decltype(tile_to_shape(
        SmemABLayoutAtom{},
        make_shape(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));
    // SmemBLayout [w_bit * block_n, block_k, stage]
    using SmemBLayout = decltype(tile_to_shape(
        SmemABLayoutAtom{},
        make_shape(Int<MainLoop_BLOCK_N>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));

    static constexpr size_t ASmemSize = cosize(SmemALayout{}) / 8; // bit pack
    static constexpr size_t BSmemSize = cosize(SmemBLayout{}) / 8; // bit pack
    static constexpr size_t inputSmemSize = ASmemSize + BSmemSize;

    // SmemCLayout [x_bit * block_m, w_bit * block_n]

    // using SmemR2SCLayout =
    //     // decltype(composition(C_Swizzle{}, make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
    //     //                                               make_stride(Int<BLOCK_N>{}, Int<1>{}))));
    //     decltype(make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));
    // using SmemCLayoutAtom = decltype(make_layout(
    //     make_shape(make_shape(_4{}, Int<BLOCK_M / 4>{}), make_shape(_8{}, Int<BLOCK_N / 8>{})),
    //     make_stride(make_stride(_8{}, Int<32 * BLOCK_N / 8>{}), make_stride(_1{}, _32{}))));
    using SmemCLayout = decltype(make_layout(
        make_shape(make_shape(_4{}, Int<BLOCK_M / 4>{}, Int<X_BITS>{}),
                   make_shape(_8{}, Int<BLOCK_N / 8>{}, Int<W_BITS>{})),
        make_stride(make_stride(_8{}, Int<4 * BLOCK_N>{}, Int<BLOCK_M * BLOCK_N * W_BITS>{}),
                    make_stride(_1{}, _32{}, Int<BLOCK_M * BLOCK_N>{}))));

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
    // int main_loop_m = M * X_BITS;
    // int main_loop_n = N * W_BITS;
    int main_loop_k = K;
    // gmem tensor
    // auto A_tensor = make_tensor(make_gmem_ptr<type>(X), make_shape(main_loop_m, main_loop_k),
    //                             make_stride(main_loop_k, _1{}));
    // auto B_tensor = make_tensor(make_gmem_ptr<type>(W), make_shape(main_loop_n, main_loop_k),
    //                             make_stride(main_loop_k, _1{}));
    auto A_tensor =
        make_tensor(make_gmem_ptr<type>(X), make_shape(make_shape(M, Int<X_BITS>{}), main_loop_k),
                    make_stride(make_stride(main_loop_k, M * main_loop_k), _1{}));
    auto B_tensor =
        make_tensor(make_gmem_ptr<type>(W), make_shape(make_shape(N, Int<W_BITS>{}), main_loop_k),
                    make_stride(make_stride(main_loop_k, N * main_loop_k), _1{}));
    // pred tensor for OOB(out of bound) check
    auto A_pred_tensor = make_identity_tensor(shape(A_tensor));
    auto B_pred_tensor = make_identity_tensor(shape(B_tensor));

    // gmem tile tensor
    auto gA =
        local_tile(A_tensor,
                   make_tile(make_tile(Int<BLOCK_M>{}, Int<X_BITS>{}), Int<MainLoop_BLOCK_K>{}),
                   make_coord(bidx_m, _)); //[(block_m, P), block_k, k_loop]
    auto gB =
        local_tile(B_tensor,
                   make_tile(make_tile(Int<BLOCK_N>{}, Int<W_BITS>{}), Int<MainLoop_BLOCK_K>{}),
                   make_coord(bidx_n, _)); //[(block_n, Q), block_k, k_loop]
    auto gA_pred =
        local_tile(A_pred_tensor,
                   make_tile(make_tile(Int<BLOCK_M>{}, Int<X_BITS>{}), Int<MainLoop_BLOCK_K>{}),
                   make_coord(bidx_m, _)); //[(block_m, P), block_k, k_loop]
    auto gB_pred =
        local_tile(B_pred_tensor,
                   make_tile(make_tile(Int<BLOCK_N>{}, Int<W_BITS>{}), Int<MainLoop_BLOCK_K>{}),
                   make_coord(bidx_m, _)); //[(block_n, Q), block_k, k_loop]

    // smem tile tensor
    type *a_smem_ptr = reinterpret_cast<type *>(shared_mem_workspace);

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
    auto tAgA_g2s_copy_pred =
        a_thr_g2s_copy.partition_S(gA_pred); // [copy_size, copy_m, copy_k, k_loop]
    auto tAsA_g2s_copy = a_thr_g2s_copy.partition_D(sA); // [copy_size, copy_m, copy_k, stage]

    auto b_thr_g2s_copy = b_g2s_copy.get_slice(tidx);
    auto tBgB_g2s_copy = b_thr_g2s_copy.partition_S(gB); // [copy_size, copy_n, copy_k, k_loop]
    auto tBgB_g2s_copy_pred =
        b_thr_g2s_copy.partition_S(gB_pred); // [copy_size, copy_n, copy_k, k_loop]
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
    if (thread(tid)) {
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
        print("\ngA\n");
        print(gA);
        print("\ngB\n");
        print(gB);
        print("\nsA\n");
        print(sA);
        print("\nsB\n");
        print(sB);
        print("\ngA_pred\n");
        print(gA_pred);
        print("\ngB_pred\n");
        print(gB_pred);
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
    //     // main loop
    const int k_main_loop_cnt = size<2>(gA);
    const int k_inner_loop_cnt = size<2>(tArA_mma);
    int g2s_s_write_cnt = 0;
    int g2s_g_read_cnt = 0;
    int s2r_s_read_cnt = 0;
    int next_s2r_s_read_cnt = 0;
    auto m_tile_bound = make_tuple((bidx_m + 1) * BLOCK_M, Int<X_BITS>{});
    auto n_tile_bound = make_tuple((bidx_n + 1) * BLOCK_N, Int<W_BITS>{});
    // g2s pipeline
#pragma unroll
    for (int i_stage = 0; i_stage < kStage - 1; i_stage++) {
        auto a_tile_bound = make_tuple(m_tile_bound, (i_stage + 1) * MainLoop_BLOCK_K);
        auto b_tile_bound = make_tuple(n_tile_bound, (i_stage + 1) * MainLoop_BLOCK_K);
        if (g2s_g_read_cnt < k_main_loop_cnt) {
            copy_strip_zfill(a_g2s_copy, tAgA_g2s_copy_pred(_, _, _, i_stage),
                             tAgA_g2s_copy(_, _, _, i_stage), tAsA_g2s_copy(_, _, _, i_stage),
                             a_tile_bound, shape(A_tensor));
            __syncwarp(); // reduce random store bank conflict 
            copy_strip_zfill(b_g2s_copy, tBgB_g2s_copy_pred(_, _, _, i_stage),
                             tBgB_g2s_copy(_, _, _, i_stage), tBsB_g2s_copy(_, _, _, i_stage),
                             b_tile_bound, shape(B_tensor));
            __syncwarp();
        }
        g2s_g_read_cnt++;
        g2s_s_write_cnt++;
        cp_async_fence();
    }
    // cp_async_wait<kStage - 2>();
    // __syncthreads();
    // if (thread(tid)) {
    //     print("\nsA\n");
    //     print_tensor(sA);
    //     print("\nBA\n");
    //     print_tensor(sB);
    // }
    // __syncthreads();

    // enable s2r register pipeline when k_inner_loop_cnt > 1
    if (k_inner_loop_cnt > 1) {
        // wait first cp_async commit
        cp_async_wait<kStage - 2>();
        __syncthreads();
        // load first s2r
        copy(a_s2r_copy, tAsA_s2r_copy(_, _, 0, s2r_s_read_cnt), tArA_s2r_copy(_, _, 0));
        copy(b_s2r_copy, tBsB_s2r_copy(_, _, 0, s2r_s_read_cnt), tBrB_s2r_copy(_, _, 0));
    }

#pragma unroll
    for (int k_main_loop_idx = 0; k_main_loop_idx < k_main_loop_cnt; k_main_loop_idx++) {
#pragma unroll
        for (int k_inner_loop_idx = 0; k_inner_loop_idx < k_inner_loop_cnt; k_inner_loop_idx++) {
            int next_k_inner_loop_idx = (k_inner_loop_idx + 1) % k_inner_loop_cnt;
            // wait next stage commit
            if (k_inner_loop_idx == k_inner_loop_cnt - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                s2r_s_read_cnt = next_s2r_s_read_cnt;
                // s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
            }
            // s2r pipeline
            copy(a_s2r_copy, tAsA_s2r_copy(_, _, next_k_inner_loop_idx, s2r_s_read_cnt),
                 tArA_s2r_copy(_, _, next_k_inner_loop_idx));
            copy(b_s2r_copy, tBsB_s2r_copy(_, _, next_k_inner_loop_idx, s2r_s_read_cnt),
                 tBrB_s2r_copy(_, _, next_k_inner_loop_idx));
            // load last stage
            if (k_inner_loop_idx == 0) {
                auto a_tile_bound =
                    make_tuple(m_tile_bound, (g2s_g_read_cnt + 1) * MainLoop_BLOCK_K);
                auto b_tile_bound =
                    make_tuple(n_tile_bound, (g2s_g_read_cnt + 1) * MainLoop_BLOCK_K);
                // OOB do not g2s copy
                if (g2s_g_read_cnt < k_main_loop_cnt) {
                    copy_strip_zfill(a_g2s_copy, tAgA_g2s_copy_pred(_, _, _, g2s_g_read_cnt),
                                     tAgA_g2s_copy(_, _, _, g2s_g_read_cnt),
                                     tAsA_g2s_copy(_, _, _, g2s_s_write_cnt), a_tile_bound,
                                     shape(A_tensor));
                    __syncwarp();
                    copy_strip_zfill(b_g2s_copy, tBgB_g2s_copy_pred(_, _, _, g2s_g_read_cnt),
                                     tBgB_g2s_copy(_, _, _, g2s_g_read_cnt),
                                     tBsB_g2s_copy(_, _, _, g2s_s_write_cnt), b_tile_bound,
                                     shape(B_tensor));
                    __syncwarp();
                }
                g2s_g_read_cnt++;
                g2s_s_write_cnt = s2r_s_read_cnt;
                next_s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
                cp_async_fence();
            }
            // gemm
            gemm(mma, tArA_mma(_, _, k_inner_loop_idx), tBrB_mma(_, _, k_inner_loop_idx), tCrC_mma);
        }
    }

    copy(c_r2s_copy, tCrC_r2s_copy, tCsS_r2s_copy); 
    __syncthreads();

#if 0 // check mainloop result
    if (thread(tid)) {
        print("\ntid:%d\n", threadIdx.x);
        print("\ntCrC\n");
        print_tensor(tCrC_mma);
        print("\nsC\n");
        print_tensor(sC);
    }
    __syncthreads();
#endif
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
    auto D_pred_tensor = make_identity_tensor(shape(D_tensor));

    auto gD = local_tile(D_tensor, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                         make_coord(bidx_m, bidx_n)); //[block_m, block_n]

    auto gD_pred = local_tile(D_pred_tensor, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                              make_coord(bidx_m, bidx_n)); //[block_m, block_n]

    auto sC = make_tensor(make_smem_ptr<acc_type>(shared_mem_workspace),
                          SmemCLayout{}); // [P * block_m, Q * block_n]
    auto sC_pred = make_identity_tensor(shape(sC));

    auto sC_tiled = local_tile(sC, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                               make_coord(_, _)); // [block_m, block_n, P, Q]

    auto sC_tiled_pred = local_tile(sC_pred, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                                    make_coord(_, _)); // [block_m, block_n, P, Q]

    // Epilog s2r copy
    EpilogS2RCopy c_s2r_copy;
    auto c_thr_s2r_copy = c_s2r_copy.get_slice(tidx);
    auto tCsC_s2r_copy = c_thr_s2r_copy.partition_S(sC_tiled); // [copy_size, copy_m, copy_n, P, Q]
    auto tCsC_s2r_copy_pred =
        c_thr_s2r_copy.partition_S(sC_tiled_pred); // [copy_size, copy_m, copy_n, P, Q]
    auto tCrC_s2r_copy =
        make_tensor_like(tCsC_s2r_copy(_, _, _, 0, 0)); // [copy_size, copy_m, copy_n]
    auto tCrC_reduce = make_tensor_like(tCrC_s2r_copy);

    // Epilog r2g copy
    EpilogR2GCopy c_r2g_copy;
    auto c_thr_r2g_copy = c_r2g_copy.get_slice(tidx);
    auto tCrC_r2g_copy = c_thr_r2g_copy.retile_S(tCrC_reduce);
    auto tCgC_r2g_copy = c_thr_r2g_copy.partition_D(gD);
    auto tCgC_r2g_copy_pred = c_thr_r2g_copy.partition_D(gD_pred);

#if 0 // print for debug
    if (thread(tid)) {
        print("\nc_s2r_copy\n");
        print(c_s2r_copy);
        print("\nc_r2g_copy\n");
        print(c_r2g_copy);
        print("\nsC\n");
        print(sC);
        print("\nsC_pre\n");
        print(sC_pred);
        print("\nsC_tiled\n");
        print(sC_tiled);
        print("\ntCsC_s2r_copy\n");
        print(tCsC_s2r_copy);
        print("\ntCsC_s2r_copy_pred\n");
        print(tCsC_s2r_copy_pred);
        print("\ntCrC_s2r_copy\n");
        print(tCrC_s2r_copy);
        print("\ntCrC_r2g_copy\n");
        print(tCrC_r2g_copy);
        print("\ntCgC_r2g_copy\n");
        print(tCgC_r2g_copy);
        print("\ntCgC_r2g_copy_pred\n");
        print(tCgC_r2g_copy_pred);
    }
#endif
    // epilogue
    auto epilog_bound = make_tuple(make_tuple(Int<4>{}, Int<BLOCK_M / 4>{}, _1{}),
                                   make_tuple(Int<8>{}, Int<BLOCK_N / 8>{}, _1{}));
    // if (thread(tid)) {
    //     print("\nepilog\n");
    //     print(epilog_bound);
    //     print("\npred\n");
    //     print(tCsC_s2r_copy_pred(_0{}));
    //     print("\ncheck:%d\n", elem_less(tCsC_s2r_copy_pred(_0{}), epilog_bound));
    // }

    // thread is valid that pred in [block_m, block_n] range
    if (elem_less(tCsC_s2r_copy_pred(_0{}), epilog_bound)) {
        acc_type multiplier = 1;
#pragma unroll
        for (int x_bit_idx = 0; x_bit_idx < X_BITS; x_bit_idx++) {
            acc_type cur_multiplier =
                (quant_signed && (x_bit_idx == X_BITS - 1)) ? -1 * multiplier : multiplier;
#pragma unroll
            for (int w_bit_idx = 0; w_bit_idx < W_BITS; w_bit_idx++) {
                copy(c_s2r_copy, tCsC_s2r_copy(_, _, _, x_bit_idx, w_bit_idx), tCrC_s2r_copy);
                // if (thread(tid)) {
                //     print("\ns2r_copy\n");
                //     print(tCrC_s2r_copy);
                // }
#pragma unroll
                for (int i = 0; i < size(tCrC_s2r_copy); i++) {
                    tCrC_reduce(i) += cur_multiplier * tCrC_s2r_copy(i);
                }
                cur_multiplier = (quant_signed && (w_bit_idx == W_BITS - 2)) ? -2 * cur_multiplier :
                                                                               2 * cur_multiplier;
            }
            multiplier <<= 1;
        }
        //r2g store
        // copy(c_r2g_copy, tCrC_r2g_copy, tCgC_r2g_copy);
        // if (thread(tid)) {
        //     print(tCrC_r2g_copy);
        //     print("\n");
        //     print(tCgC_r2g_copy_pred(_0{}));
        // }
        copy_if(
            c_r2g_copy,
            [&](auto... coords) {
                return elem_less(tCgC_r2g_copy_pred(coords...), shape(D_tensor));
            },
            tCrC_r2g_copy, tCgC_r2g_copy);
    }
}