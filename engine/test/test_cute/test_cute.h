

#pragma once
#include <string>
#include <cuda_runtime.h>
#include "common/pack.h"
#include "common/timer.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/tensor_view_io.h"
#include "cute/tensor.hpp"
#define TEST(X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE)          \
    {                                                                                              \
        std::cout << "SM:" << GPU_ARCH << " Config:" << config_str << " ";                         \
        printf("BM:%4d,BN:%4d,BK:%4d,WM:%4d,WN:%4d,WK:%4d,MMA_N:%4d,MMA_N%4d,MMA_K%4d,STAGE:%4d|", \
               BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE);                               \
        int ret = benchmark<AQ_INIT_FUN(AqCute), AQ_EXEC_FUN(AqCute), AQ_OP_STATE(AqCute)>(        \
            AQ_NAME_FUN(AqCute, Init, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,       \
                        MMA_N, MMA_K, NSTAGE),                                                     \
            AQ_NAME_FUN(AqCute, Exec, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,       \
                        MMA_N, MMA_K, NSTAGE),                                                     \
            x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, nullptr, h_out,          \
            h_ref_out, false, SIGNED, exec_dur, pack_dur, stream, warmup, repeat);                 \
        if (ret == 0 && gflop_count / exec_dur > max_gflop) {                                      \
            max_gflop = gflop_count / exec_dur;                                                    \
            max_bw = gbyte_count / (exec_dur * 1e-3);                                              \
            best_config.str("");                                                                   \
            best_config << BM << ", " << BN << ", " << BK << ", " << WM << ", " << WN << ", "      \
                        << WK << ", " << MMA_M << ", " << MMA_N << ", " << MMA_K << ", "           \
                        << NSTAGE;                                                                 \
        }                                                                                          \
        printf("packing %f (us) exec %f (us)| %f GBPS | %f TOPS | %f B-TOPS | %s\n",               \
               pack_dur * 1e3, exec_dur * 1e3, gbyte_count / (exec_dur * 1e-3),                    \
               gflop_count / exec_dur, true_gflop_count / exec_dur,                                \
               ret == 0  ? "PASSED" :                                                              \
               ret == -1 ? "ERROR" :                                                               \
                           "FAILED");                                                              \
    }

inline bool isCudaSuccess(cudaError_t status)
{
    cudaError_t error = status;
    if (error != cudaSuccess) {
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

inline bool check(const int *ref_out, const int *out, int m, int n)
{
    for (int i = 0; i < m * n; ++i) {
        if (ref_out[i] != out[i]) {
            return false;
        }
    }
    return true;
}

/// benchmark func for cute
template <typename InitFuncType, typename ExecFuncType, typename OpStateType>
inline int benchmark(InitFuncType init_fn, ExecFuncType exec_fn, int X_BITS, int W_BITS, int *X,
                     int *W, int *X_PACKED, int *W_PACKED, int M, int N, int K, int *D, half *C,
                     int *H_OUT, const int *H_REF_OUT, bool bias, bool SIGNED, float &exec_dur,
                     float &pack_dur, cudaStream_t stream = NULL, int warmup = 0, int repeat = 1)
{
    auto w_pack_func = [&]() {
        if (W_BITS <= 32) {
            cudaError_t err = launch_pack(W, W_PACKED, N, K, W_BITS, stream);
            if (err != cudaSuccess) {
                printf("Line %d: 'weight launch_pack' failed: %s\n", __LINE__,
                       cudaGetErrorString(err));
            }
        } else {
            printf("unsupport W_BITS %d: for launch_pack func \n", W_BITS);
            return -1;
        }
        return 0;
    };

    auto x_pack_func = [&]() {
        if (X_BITS <= 32) {
            cudaError_t err = launch_pack(X, X_PACKED, M, K, X_BITS, stream);
            if (err != cudaSuccess) {
                printf("Line %d: 'activation launch_pack' failed: %s\n", __LINE__,
                       cudaGetErrorString(err));
            }
        } else {
            printf("unsupport X_BITS %d: for launch_pack func \n", X_BITS);
            return -1;
        }
        return 0;
    };

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }
    // w_pack_func();
    // x_pack_func();
    // cudaDeviceSynchronize();
    // if (cudaGetLastError() != cudaSuccess) {
    //     std::cerr << "return due to previous error. ";
    //     return -1;
    // }

    // {
    //     /*for debug mainloop*/
    //     using namespace cute;
    //     auto mnk = cutlass::gemm::GemmCoord({ X_BITS * M, W_BITS * N, K });
    //     cutlass::HostTensor<cutlass::uint1b_t, cutlass::layout::RowMajor> A(
    //         cutlass::MatrixCoord({ X_BITS * M, K }));
    //     cutlass::HostTensor<cutlass::uint1b_t, cutlass::layout::ColumnMajor> B(
    //         cutlass::MatrixCoord({
    //             K,
    //             W_BITS * N,
    //         }));
    //     cutlass::HostTensor<cutlass::uint1b_t, cutlass::layout::RowMajor> B_row(
    //         cutlass::MatrixCoord({ W_BITS * N, K }));

    //     cutlass::HostTensor<int32_t, cutlass::layout::RowMajor> C({ X_BITS * M, W_BITS * N });

    //     cudaMemcpy(A.device_data(), X_PACKED, X_BITS * M * K / 8, cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(B.device_data(), W_PACKED, W_BITS * N * K / 8, cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(B_row.device_data(), W_PACKED, W_BITS * N * K / 8, cudaMemcpyDeviceToDevice);

    //     A.sync_host();
    //     B.sync_host();
    //     B_row.sync_host();
    //     auto A_tensor = make_tensor(make_gmem_ptr<uint1_t>(A.host_data()),
    //                                 make_shape(mnk.m(), mnk.k()), make_stride(mnk.k(), _1{}));
    //     // std::cout << std::endl << B_row.host_view() << std::endl;

    //     auto B_tensor = make_tensor(make_gmem_ptr<uint1_t>(B.host_data()),
    //                                 make_shape(W_BITS * N, K), make_stride(K, _1{}));
    //     auto B_r_tensor = make_tensor(make_gmem_ptr<uint1_t>(B_row.host_data()),
    //                                   make_shape(W_BITS * N, K), make_stride(K, _1{}));
    //     using Host_GeMM =
    //         cutlass::reference::host::Gemm<cutlass::uint1b_t, cutlass::layout::RowMajor,
    //                                        cutlass::uint1b_t, cutlass::layout::ColumnMajor, int32_t,
    //                                        cutlass::layout::RowMajor, int32_t, int32_t>;
    //     Host_GeMM host_kernel;
    //     host_kernel(mnk, int32_t(1), A.host_ref(), B.host_ref(), int32_t(0), C.host_ref());
    //     // std::cout << "C_ref\n";
    //     // std::cout << C.host_view() << std::endl;
    //     auto C_tensor = make_tensor(make_gmem_ptr<int32_t>(C.host_data()),
    //                                 make_shape(mnk.m(), mnk.n()), make_stride(mnk.n(), _1{}));
    //     print("\nA\n");
    //     print_tensor(A_tensor);
    //     print("\nB\n");
    //     print_tensor(B_tensor);
    //     // print("\nB_r\n");
    //     // print_tensor(B_r_tensor);
    //     print("\ngemm ref\n");
    //     print_tensor(C_tensor);
    // }

    OpStateType state = (*init_fn)(X_PACKED, W_PACKED, M, N, K, D, nullptr, false);
    if (!state.initSuccess) {
        std::cerr << "return due to unsuccessful initialization. " << std::endl;
        return -1;
    }
    // (*exec_fn)(state, stream);
    // cudaDeviceSynchronize();
    if (auto err = cudaGetLastError(); !isCudaSuccess(err)) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }

    // profling exec func
    CudaTimer exec_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            exec_timer.start();
        (*exec_fn)(state, stream);
    }
    exec_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "exec kernel failed." << std::endl;
        return -1;
    }
    exec_dur = exec_timer.elapsed_msecs() / repeat;

    // profling packing func
    CudaTimer packing_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            packing_timer.start();
        x_pack_func();
    }
    packing_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "packing kernel failed." << std::endl;
        return -1;
    }
    pack_dur = packing_timer.elapsed_msecs() / repeat;
    cudaDeviceSynchronize();
    // accuracy comparison
    cudaMemcpy(H_OUT, D, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (!check(H_REF_OUT, H_OUT, M, N)) {
        // using namespace cute;
        // Tensor ref =
        //     make_tensor(make_gmem_ptr<int32_t>(H_REF_OUT), make_shape(M, N), make_stride(N, _1{}));
        // Tensor dev =
        //     make_tensor(make_gmem_ptr<int32_t>(H_OUT), make_shape(M, N), make_stride(N, _1{}));
        // print("\nref\n");
        // print_tensor(ref);
        // print("\ndev\n");
        // print_tensor(dev);
        return -2;
    }
    return 0;
}

void test_cute_w2a2(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w2a4(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w2a6(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w2a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w3a3(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w3a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w4a4(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w4a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w5a5(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w6a6(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w7a7(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_cute_w8a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);