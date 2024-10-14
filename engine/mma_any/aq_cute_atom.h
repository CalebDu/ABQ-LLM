#pragma once

#include "common/base.h"
#include "cute/arch/mma_sm80.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/math.hpp"
#include "cute/tensor.hpp"
#include "cute/swizzle.hpp"
#include "cute/swizzle_layout.hpp"

using namespace cute;

template <typename MmaShape> struct AqCuteMmaAtom;
// mma_op m8n8k128
template <> struct AqCuteMmaAtom<ShapeBase<8, 8, 128>> {
    using mma_op = cute::SM80_8x8x128_S32U1U1S32_TN_XORPOPC;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
};
// mma_op m16n8k128
template <> struct AqCuteMmaAtom<ShapeBase<16, 8, 128>> {
    using mma_op = cute::SM80_16x8x128_S32U1U1S32_TN_XORPOPC;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
};
// mma_op m16n8k256
template <> struct AqCuteMmaAtom<ShapeBase<16, 8, 256>> {
    using mma_op = cute::SM80_16x8x256_S32U1U1S32_TN_XORPOPC;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
};

// define all SmemLayout in abq kernel
template <typename QuantType, typename ThreadBlockShape, int kThreadBlockStage>
struct AqCuteSmemLayout {
    static constexpr int X_BITS = QuantType::X_BITS;
    static constexpr int W_BITS = QuantType::W_BITS;
    static constexpr int G2S_SwizzleB = 3; // 2^3 = 8
    static constexpr int G2S_SwizzleM = 7; // 2^7 = 128 uint1b_t = int128_t
    static constexpr int G2S_SwizzleS = 3; // 2^3 = 8
    static constexpr int BLOCK_M = ThreadBlockShape::M;
    static constexpr int BLOCK_N = ThreadBlockShape::N;
    static constexpr int BLOCK_K = ThreadBlockShape::K;

    static constexpr int MainLoop_BLOCK_M = BLOCK_M * X_BITS;
    static constexpr int MainLoop_BLOCK_N = BLOCK_N * W_BITS;
    static constexpr int MainLoop_BLOCK_K = BLOCK_K;

    // mainloop SmemLayout
    using SmemABLayoutAtom =
        decltype(composition(Swizzle<G2S_SwizzleB, G2S_SwizzleM, G2S_SwizzleS>{},
                             make_layout(make_shape(Int<8>{}, Int<MainLoop_BLOCK_K>{}),
                                         make_stride(Int<MainLoop_BLOCK_K>{}, Int<1>{}))));
    // [x_bit * block_m, K]
    using SmemALayout = decltype(tile_to_shape(
        SmemABLayoutAtom{},
        make_shape(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));
    // [w_bit * block_n, K]
    using SmemBLayout = decltype(tile_to_shape(
        SmemABLayoutAtom{},
        make_shape(Int<MainLoop_BLOCK_N>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));
    static constexpr size_t inputSmemSize = cosize(SmemALayout{}) + cosize(SmemBLayout{});

    // todo SmemCLayout
    // [x_bit * block_m, w_bit * block_n]

    // epilogue SmemLayout
    // todo SmemEpilogStore
    // [block_m, block_n]

    static constexpr size_t SmemSize = inputSmemSize;
};

// define all copy in abq kernel
template <typename type, typename ThreadBlockShape, int kThread> struct AqCuteCopy {
    //  mainloop G2S load async copy
    using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>; // 128 uint1b_t per thread
    using G2SCopyTraits = Copy_Traits<G2SCopyOp>;
    using G2SCopyAtom = Copy_Atom<G2SCopyTraits, type>;

    static constexpr int G2SCopy_thread_k = ThreadBlockShape::K / 128;
    static constexpr int G2SCopy_thread_m = kThread / G2SCopy_thread_k;

    // copy tile: thread layout[G2SCopy_thread_m, G2SCopy_thread_k] async copy uint128_t per thread
    using G2SCopyA = decltype(make_tiled_copy(
        G2SCopyAtom{},
        make_layout(make_shape(Int<G2SCopy_thread_m>{}, Int<G2SCopy_thread_k>{}),
                    make_stride(Int<G2SCopy_thread_k>{}, _1{})),
        make_layout(make_shape(_1{}, _128{}))));
    using G2SCopyB = G2SCopyA;

    // mainloop S2R load copy
    // copy 32 uint1b_t[fragment] per thread
    using S2RCopyOp = UniversalCopy<int32_t>;
    using S2RCopyTraits = Copy_Traits<S2RCopyOp>;
    using S2RCopyAtom = Copy_Atom<S2RCopyTraits, type>;
    using S2RCopyAtomA = S2RCopyAtom;
    using S2RCopyAtomB = S2RCopyAtom;

    // mainloop R2S store copy
    //todo

    // epilogue S2R load copy
    //todo

    // epilogue R2S store copy
    //todo

    // epilogue S2G store copy vectorization
    //todo
};