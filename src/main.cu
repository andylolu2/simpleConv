#include <cute/tensor.hpp>

#include "src/conv.cuh"

namespace ct = cute;
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

void set_arange(ct::half_t *data, int64_t size) {
    std::vector<ct::half_t> host_data(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data[i] = static_cast<ct::half_t>(static_cast<float>(i) / 64);
    }
    CUTE_CHECK_ERROR(cudaMemcpy(data, host_data.data(), size * sizeof(ct::half_t), cudaMemcpyHostToDevice));
}

int main(int argc, char const *argv[]) {
    int64_t N = 32;
    int64_t H = 32;
    int64_t W = 32;
    int64_t C = 128;
    int64_t K = 128;
    int64_t R = 3;
    int64_t S = 3;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t dilation_h = 1;
    int64_t dilation_w = 1;
    int64_t P = ((H + 2 * pad_h - R * dilation_h) / stride_h) + 1;
    int64_t Q = ((W + 2 * pad_w - S * dilation_w) / stride_w) + 1;

    // int64_t N = 1;
    // int64_t H = 16;
    // int64_t W = 16;
    // int64_t C = 64;
    // int64_t K = 128;
    // int64_t R = 3;
    // int64_t S = 3;

    ct::half_t *data_img;
    ct::half_t *data_kernel;
    ct::half_t *data_out;
    CUTE_CHECK_ERROR(cudaMalloc(&data_img, N * H * W * C * sizeof(ct::half_t)));
    CUTE_CHECK_ERROR(cudaMalloc(&data_kernel, K * R * S * C * sizeof(ct::half_t)));
    CUTE_CHECK_ERROR(cudaMalloc(&data_out, N * P * Q * K * sizeof(ct::half_t)));
    set_arange(data_img, N * H * W * C);
    set_arange(data_kernel, K * R * S * C);
    set_arange(data_out, N * P * Q * K);

    auto img = ct::make_tensor(ct::make_gmem_ptr(data_img), ct::make_shape(N, H, W, C), ct::GenRowMajor{});
    auto kernel = ct::make_tensor(ct::make_gmem_ptr(data_kernel), ct::make_shape(K, R, S, C), ct::GenRowMajor{});
    auto out = ct::make_tensor(ct::make_gmem_ptr(data_out), ct::make_shape(N, P, Q, K), ct::GenRowMajor{});

    conv2d(img, kernel, out);

    cudaDeviceSynchronize();

    CUTE_CHECK_ERROR(cudaFree(data_img));
    CUTE_CHECK_ERROR(cudaFree(data_kernel));
    CUTE_CHECK_ERROR(cudaFree(data_out));

    return 0;
}
