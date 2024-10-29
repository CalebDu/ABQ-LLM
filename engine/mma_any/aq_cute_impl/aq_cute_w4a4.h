#pragma once
#include "common/base.h"
#include "mma_any/aq_cute_op.h"


#ifdef W4A4
AQ_DECL_FUN(AqCute, 4, 4, true, 2, 64, 1024, 1, 4, 1, 8, 8, 128, 3);
AQ_DECL_FUN(AqCute, 4, 4, true, 2, 64, 512, 1, 4, 1, 8, 8, 128, 6);

AQ_DECL_FUN(AqCute, 4, 4, true, 2, 32, 512, 1, 4, 1, 8, 8, 128, 8);
AQ_DECL_FUN(AqCute, 4, 4, true, 2, 32, 1024, 1, 4, 1, 8, 8, 128, 4);
#endif