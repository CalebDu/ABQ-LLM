
#include "common/base.h"
#include "mma_any/aq_cute_op.h"

// cta<8,32,128>  warp layout<1,2,1> mma<8,8,128>  stage2
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 2);

// // cta<8,32,128>  warp layout<1,2,1> mma<8,8,128>  stage3
// AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 3);

// // cta<8,32,128>  warp layout<1,2,1> mma<8,8,128>  stage4
// AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 4);

// // cta<8,32,128>  warp layout<1,2,1> mma<8,8,128>  stage5
// AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 5);