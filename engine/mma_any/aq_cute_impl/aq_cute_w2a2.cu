#include "common/base.h"
#include "mma_any/aq_cute_op.h"

#ifdef W2A2
// cta<8,32,128>  warp layout<1,2,1> mma<8,8,128> stage2
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 5);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 6);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 7);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 8);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 9);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 8, 32, 128, 1, 2, 1, 8, 8, 128, 10);

// cta<4,32,128>  warp layout<1,2,1> mma<8,8,128>  stage2
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 5);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 6);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 7);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 8);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 9);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 128, 1, 2, 1, 8, 8, 128, 10);

// cta<4,64,128>  warp layout<1,2,1> mma<8,8,128>  stage2
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 5);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 6);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 7);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 8);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 9);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 2, 1, 8, 8, 128, 10);

// cta<4,64,128>  warp layout<1,4,1> mma<8,8,128>  stage2
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 5);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 6);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 7);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 8);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 9);
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 128, 1, 4, 1, 8, 8, 128, 10);
// cta<4,64,256>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 256, 1, 4, 1, 8, 8, 128, 10);

// cta<4,64,384>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 384, 1, 4, 1, 8, 8, 128, 10);

// cta<4,64,512>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 512, 1, 4, 1, 8, 8, 128, 10);

// cta<4,64,512>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 64, 1024, 1, 4, 1, 8, 8, 128, 5);

// cta<4,32,256>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 256, 1, 4, 1, 8, 8, 128, 10);

// cta<4,32,384>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 384, 1, 4, 1, 8, 8, 128, 10);

// cta<4,32,512>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 512, 1, 4, 1, 8, 8, 128, 10);

// cta<4,32,512>  warp layout<1,4,1> mma<8,8,128>
AQ_INSTANTIATE_FUN(AqCute, 2, 2, true, 4, 32, 1024, 1, 4, 1, 8, 8, 128, 5);
#endif