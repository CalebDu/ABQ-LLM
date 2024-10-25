#pragma once
#include "common/base.h"
#include "mma_any/aq_cute_op.h"

#ifdef W2A2
// cta<8,32,128>  warp layout<1,2,1> mma<8,8,128> stage2
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 2);

// cta<8,32,128>  warp layout<1,2,1> mma<8,8,128> stage3
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 3);

// cta<8,32,128>  warp layout<1,2,1> mma<8,8,128> stage4
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 4);

// cta<8,32,128>  warp layout<1,2,1> mma<8,8,128> stage5
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 5);

// cta<4,32,128>  warp layout<1,2,1> mma<8,8,128>  stage2
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 2);
// cta<4,32,128>  warp layout<1,2,1> mma<8,8,128>  stage3
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 3);
// cta<4,32,128>  warp layout<1,2,1> mma<8,8,128>  stage4
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 4);
// cta<4,32,128>  warp layout<1,2,1> mma<8,8,128>  stage5
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 5);

// cta<4,64,128>  warp layout<1,2,1> mma<8,8,128>  stage2
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 2);
// cta<4,64,128>  warp layout<1,2,1> mma<8,8,128>  stage3
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 3);
// cta<4,64,128>  warp layout<1,2,1> mma<8,8,128>  stage4
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 4);
// cta<4,64,128>  warp layout<1,2,1> mma<8,8,128>  stage5
AQ_DECL_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 5);
#endif