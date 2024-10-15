#pragma once
#include "aq_cute_op.h"
#include "common/base.h"

// cta<8,32,128>  warp layout<2,2,1> mma<8,8,128> 
AQ_DECL_FUN(AqCute, 2, 2, true, 8, 32, 128, 2, 2, 1, 8, 8, 128, 3);