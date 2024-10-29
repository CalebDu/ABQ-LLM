#include "common/base.h"
#include "mma_any/aq_cute_op.h"

#ifdef W3A3
AQ_INSTANTIATE_FUN(AqCute, 3, 3, true, 2, 64, 1024, 1, 4, 1, 8, 8, 128, 3);

AQ_INSTANTIATE_FUN(AqCute, 3, 3, true, 2, 32, 512, 1, 4, 1, 8, 8, 128, 8);
#endif