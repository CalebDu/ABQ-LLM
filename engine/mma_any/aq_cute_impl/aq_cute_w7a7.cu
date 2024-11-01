#include "common/base.h"
#include "mma_any/aq_cute_op.h"

#ifdef W7A7

AQ_INSTANTIATE_FUN(AqCute, 7, 7, true, 1, 64, 512, 1, 4, 1, 8, 8, 128, 3);

AQ_INSTANTIATE_FUN(AqCute, 7, 7, true, 1, 32, 512, 1, 4, 1, 8, 8, 128, 6);
AQ_INSTANTIATE_FUN(AqCute, 7, 7, true, 1, 32, 1024, 1, 4, 1, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqCute, 7, 7, true, 1, 32, 512, 1, 4, 1, 8, 8, 128, 3);
#endif