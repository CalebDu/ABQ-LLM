#include "common/base.h"
#include "mma_any/aq_cute_op.h"

#ifdef W6A6
AQ_INSTANTIATE_FUN(AqCute, 6, 6, true, 1, 64, 1024, 1, 4, 1, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqCute, 6, 6, true, 1, 64, 512, 1, 4, 1, 8, 8, 128, 4);

AQ_INSTANTIATE_FUN(AqCute, 6, 6, true, 1, 32, 512, 1, 4, 1, 8, 8, 128, 7);
AQ_INSTANTIATE_FUN(AqCute, 6, 6, true, 1, 32, 1024, 1, 4, 1, 8, 8, 128, 3);

AQ_INSTANTIATE_FUN(AqCute, 6, 6, true, 1, 32, 512, 1, 4, 1, 8, 8, 128, 4);
#endif