

#include <string>
#include <sstream>
#include "mma_any/aq_cute_library.h"
#include "mma_any/aq_cute_op.h"
#include "test/test_cute/test_cute.h"

void test_cute_w3a3(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream)
{
#ifdef W3A3
    std::string config_str;
    std::stringstream s;
    s << "X_BITS:" << x_bits << " W_BITS:" << w_bits << " M:" << m << " N:" << n << " K:" << k
      << " ";
    if (quant_sign) {
        s << "sign ";
    } else {
        s << "unsigned ";
    }
    config_str = s.str();
    float exec_dur = 0;
    float pack_dur = 0;
    float true_gflop_count = (float)m / 1e9 * n * k * 2 * x_bits * w_bits;
    float gflop_count = (float)m / 1e9 * n * k * 2;
    float max_gflop = 0;
    float gbyte_count =
        float((x_bits * m * k + x_bits * n * k) / 8 + (m * k * sizeof(int32_t))) / 1e9;
    float max_bw = 0;
    std::stringstream best_config;

    if (quant_sign) {
        TEST(3, 3, true, 2, 64, 1024, 1, 4, 1, 8, 8, 128, 3);
        TEST(3, 3, true, 2, 32, 512, 1, 4, 1, 8, 8, 128, 8);
    } else {
    }

    printf("The best kernel config is %s with %f TOPS, BW %f GBPS\n", best_config.str().c_str(),
           max_gflop, max_bw);
#else
    printf("unsupport w%da%d\n", w_bits, x_bits);
#endif
}