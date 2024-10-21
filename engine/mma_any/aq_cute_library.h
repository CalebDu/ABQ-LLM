#pragma once
#include "aq_cute_op.h"
#include "common/base.h"

// cta<8,32,128>  warp layout<2,2,1> mma<8,8,128> stage2
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 2, 2, 1, 8, 8, 128, 2);

// cta<8,32,128>  warp layout<2,2,1> mma<8,8,128> stage3
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 2, 2, 1, 8, 8, 128, 3);

// cta<8,32,128>  warp layout<2,2,1> mma<8,8,128> stage4
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 2, 2, 1, 8, 8, 128, 4);

// cta<8,32,128>  warp layout<2,2,1> mma<8,8,128> stage5
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 2, 2, 1, 8, 8, 128, 5);