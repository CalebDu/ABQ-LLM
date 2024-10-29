#include "common/base.h"
#include "mma_any/aq_cute_op.h"

#ifdef W5A5
AQ_INSTANTIATE_FUN(AqCute, 5, 5, true, 1, 64, 1024, 1, 4, 1, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqCute, 5, 5, true, 1, 64, 512, 1, 4, 1, 8, 8, 128, 4);

AQ_INSTANTIATE_FUN(AqCute, 5, 5, true, 1, 32, 512, 1, 4, 1, 8, 8, 128, 8);
AQ_INSTANTIATE_FUN(AqCute, 5, 5, true, 1, 32, 1024, 1, 4, 1, 8, 8, 128, 4);
#endif