#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/convolution.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <cute/tensor.hpp>

#include "src/conv.cuh"

namespace ct = cute;

int main(int argc, char const *argv[]) {
    // if (argc != 4) {
    //     std::cout << "Usage: " << argv[0] << " M N K" << std::endl;
    //     return 1;
    // }
    // int64_t M = atoi(argv[1]);
    // int64_t N = atoi(argv[2]);
    // int64_t K = atoi(argv[3]);
    int64_t N = 8;
    int64_t H = 32;
    int64_t W = 16;
    int64_t C = 128;
    int64_t P = H;
    int64_t Q = W;
    int64_t K = 128;
    int64_t R = 3;
    int64_t S = 3;

    // Allocate A, B, C
    cutlass::HostTensor<ct::half_t, cutlass::layout::TensorNHWC> input_tensor({N, H, W, C});
    cutlass::HostTensor<ct::half_t, cutlass::layout::TensorNHWC> filter_tensor({K, R, S, C});
    cutlass::HostTensor<ct::half_t, cutlass::layout::TensorNHWC> output_tensor({N, P, Q, K});
    cutlass::HostTensor<ct::half_t, cutlass::layout::TensorNHWC> output_ref_tensor({N, P, Q, K});
    auto input = ct::make_tensor(ct::make_gmem_ptr(input_tensor.device_data()), ct::make_shape(N, H, W, C), ct::GenRowMajor{});
    auto filter = ct::make_tensor(ct::make_gmem_ptr(filter_tensor.device_data()), ct::make_shape(K, R, S, C), ct::GenRowMajor{});
    auto output = ct::make_tensor(ct::make_gmem_ptr(output_tensor.device_data()), ct::make_shape(N, P, Q, K), ct::GenRowMajor{});

    // Fill with random data
    cutlass::reference::device::TensorFillRandomGaussian(input_tensor.device_view(), 0);
    cutlass::reference::device::TensorFillRandomGaussian(filter_tensor.device_view(), 1);

    // Test for correctness
    // Ours
    conv2d(input, filter, output);

    // Reference
    cutlass::conv::Conv2dProblemSize problem_size(
        N, H, W, C, P, Q, K, R, S, cutlass::conv::Mode::kCrossCorrelation);

    cutlass::reference::device::Conv2dFprop<
        cutlass::half_t,
        cutlass::layout::TensorNHWC,
        cutlass::half_t,
        cutlass::layout::TensorNHWC,
        cutlass::half_t,
        cutlass::layout::TensorNHWC,
        float>(
        problem_size,
        input_tensor.device_ref(),
        filter_tensor.device_ref(),
        output_ref_tensor.device_ref(),
        output_ref_tensor.device_ref(),
        1.0f,
        0.0f);

    cudaDeviceSynchronize();

    // Copy output data to host for comparison
    output_tensor.sync_host();
    output_ref_tensor.sync_host();

    // Compare and report metrics
    int64_t rel_err_count = 0;
    int64_t abs_err_count = 0;
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t p = 0; p < P; ++p) {
            for (int64_t q = 0; q < Q; ++q) {
                for (int64_t k = 0; k < K; ++k) {
                    float c = output_tensor.host_ref().at({n, p, q, k});
                    float c_ref = output_ref_tensor.host_ref().at({n, p, q, k});
                    float diff = std::abs(c - c_ref);
                    float rel = diff / std::abs(c_ref);
                    max_abs_err = std::max(max_abs_err, diff);
                    max_rel_err = std::max(max_rel_err, rel);
                    if (diff > 0.001f) {
                        abs_err_count++;
                    }
                    if (rel > 0.01f) {
                        rel_err_count++;
                    }
                }
            }
        }
    }

    float rel_err_prop = static_cast<float>(rel_err_count) / static_cast<float>(output_ref_tensor.capacity());
    float abs_err_prop = static_cast<float>(abs_err_count) / static_cast<float>(output_ref_tensor.capacity());
    std::cout << "Max rel err: " << max_rel_err * 100 << "%" << std::endl;
    std::cout << "Rel err prop: " << rel_err_prop * 100 << "%" << std::endl;
    std::cout << "Max abs err: " << max_abs_err << std::endl;
    std::cout << "Abs err prop: " << abs_err_prop * 100 << "%" << std::endl;

    // std::cout << "Output:" << std::endl
    //           << output_tensor.host_view() << std::endl;
    // std::cout << "Output ref:" << std::endl
    //           << output_ref_tensor.host_view() << std::endl;

    return 0;
}