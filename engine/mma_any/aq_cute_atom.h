#pragma once

#include "common/base.h"
#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include <cstdint>

using namespace cute;

template <typename MmaShape> struct AqCuteMmaAtom;
// mma_op m8n8k128
template <> struct AqCuteMmaAtom<ShapeBase<8, 8, 128>> {
    using mma_op = cute::SM80_8x8x128_S32U1U1S32_TN_ANDPOPC;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
};
// mma_op m16n8k128
template <> struct AqCuteMmaAtom<ShapeBase<16, 8, 128>> {
    using mma_op = cute::SM80_16x8x128_S32U1U1S32_TN_ANDPOPC;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
};
// mma_op m16n8k256
template <> struct AqCuteMmaAtom<ShapeBase<16, 8, 256>> {
    using mma_op = cute::SM80_16x8x256_S32U1U1S32_TN_ANDPOPC;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
};

// define copy in abq kernel
template <typename type, typename acc_type, typename ThreadBlockShape, int kThread>
struct AqCuteCopy {
    // mainloop G2S load copy async
    using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>; // 128 uint1b_t per thread
    using G2SCopyTraits = Copy_Traits<G2SCopyOp>;
    using G2SCopyAtom = Copy_Atom<G2SCopyTraits, type>;

    static constexpr int G2SCopy_thread_k =
        (ThreadBlockShape::K / 128 == 1) ?
            1 :
            ROUND_UP(ThreadBlockShape::K / 128, 2); // 128 uint1b_t(int128_t) per thread

    static constexpr int G2SCopy_thread_m = kThread / G2SCopy_thread_k;
    static constexpr int G2SCopy_thread_n = kThread / G2SCopy_thread_k;

    // CTA block G2S load copy [P * block_m, block_k] tile from [P * m, k]
    using G2SCopyA = decltype(make_tiled_copy(
        G2SCopyAtom{},
        make_layout(make_shape(Int<G2SCopy_thread_m>{}, Int<G2SCopy_thread_k>{}),
                    make_stride(Int<G2SCopy_thread_k>{}, _1{})),
        make_layout(make_shape(_1{}, _128{}))));
    using G2SCopyB = decltype(make_tiled_copy(
        G2SCopyAtom{},
        make_layout(make_shape(Int<G2SCopy_thread_n>{}, Int<G2SCopy_thread_k>{}),
                    make_stride(Int<G2SCopy_thread_k>{}, _1{})),
        make_layout(make_shape(_1{}, _128{}))));

    // mainloop S2R load copy
    // load copy 32 uint1b_t(int32_t) a,b fragment per thread
    using S2RCopyOp = UniversalCopy<int32_t>;
    using S2RCopyTraits = Copy_Traits<S2RCopyOp>;
    using S2RCopyAtom = Copy_Atom<S2RCopyTraits, type>;
    using S2RCopyAtomA = S2RCopyAtom;
    using S2RCopyAtomB = S2RCopyAtom;

    // mainloop R2S store copy
    // store copy 2 int32_t(int64_t) c fragment per thread
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int64_t>, acc_type>;

    // epilogue S2R load copy vectorization
    static constexpr int Epilog_thread_n =
        ThreadBlockShape::N / 4; // 4 int_32_t(uint128_t) per thread
    static constexpr int Epilog_thread_m = kThread / Epilog_thread_n;
    // S2R load copy 4 int32 (128bit) per thread
    using EpilogS2RCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, acc_type>;

    // epilogue R2G store copy vectorization
    using EpilogR2GCopyAtom = EpilogS2RCopyAtom;

    // CTA block S2R load copy [block_m, block_n] tile from [P * block_m, Q * block_n]
    using EpilogS2RCopy = decltype(make_tiled_copy(
        EpilogS2RCopyAtom{},
        make_layout(make_shape(Int<ThreadBlockShape::M>{}, Int<Epilog_thread_n>{}), LayoutRight{}),
        make_layout(make_shape(_1{}, _4{}))));
    // CTA block S2R load copy [block_m, block_n] tile from [P * block_m, Q * block_n]
    using EpilogR2GCopy = decltype(make_tiled_copy(
        EpilogS2RCopyAtom{},
        make_layout(make_shape(Int<ThreadBlockShape::M>{}, Int<Epilog_thread_n>{}), LayoutRight{}),
        make_layout(make_shape(_1{}, _4{}))));

    // CTA block R2G store copy [block_m, block_n] tile to [m, n]
    /* deprecated EpilogS2RCopy EpilogR2GCopy
        using EpilogS2RCopy =
        decltype(make_tiled_copy(EpilogS2RCopyAtom{},
                                 make_layout(make_shape(_4{}, make_shape(_2{}, Int<kThread / 8>{})),
                                             make_stride(_2{}, make_stride(_1{}, _8{}))),
                                 make_layout(make_shape(_1{}, make_shape(_4{}, _1{})))));
        using EpilogR2GCopy =
        decltype(make_tiled_copy(EpilogS2RCopyAtom{},
                                 make_layout(make_shape(_4{}, make_shape(_2{}, Int<kThread / 8>{})),
                                             make_stride(_2{}, make_stride(_1{}, _8{}))),
                                 make_layout(make_shape(_1{}, make_shape(_4{}, _1{})))));
    */
};

template <int M, int N, int K> struct SwizzleAtom {
    constexpr static bool enable_swizzle = cutlass::is_pow2<K / 128>::value; // check K/128 is 2^n
    constexpr static int AB_Swizzle_B =
        std::conditional_t<enable_swizzle, cutlass::log2_up<K / 128>, //swizzle<n, 7, 3>
                           cutlass::log2_up<1>>::value; // swizzle<0, 7, 3> (no swizzle)
    constexpr static int AB_Swizzle_M = 7; // 2^7 = 128 uint1b(16B)
    constexpr static int AB_Swizzle_S = 3; // 2^3 = 8 bank(16B per bank)
    using AB_Swizzle = Swizzle<AB_Swizzle_B, AB_Swizzle_M, AB_Swizzle_S>;
};
