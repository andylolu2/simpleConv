#include <cute/tensor.hpp>

namespace ct = cute;
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

template <typename Engine, typename Layout>
CUTE_HOST_DEVICE auto im2col(const ct::Tensor<Engine, Layout> &x,
                             int64_t kernel_h, int64_t kernel_w
                             //  int64_t pad_h, int64_t pad_w
) {
    CUTE_STATIC_ASSERT_V(ct::rank(x.layout()) == Int<4>{});  // Input shape: (N H W C)
    // Target shape: ((N H W) (kernel_h kernel_w C))
    auto kernel_layout = ct::make_layout(
        ct::make_shape(kernel_h, kernel_w),
        ct::select<1, 2>(x.stride()));
    auto im2col_grouped_layout = ct::make_layout(x.layout(), kernel_layout);              // ((N H W C), (kernel_h kernel_w))
    auto im2col_flat_layout = im2col_grouped_layout(ct::repeat<4>(_), ct::repeat<2>(_));  // (N H W C kernel_h kernel_w)
    auto im2col_layout = ct::make_layout(
        ct::select<0, 1, 2>(im2col_flat_layout),  // (N H W)
        ct::select<4, 5, 3>(im2col_flat_layout)   // (kernel_h kernel_w C)
    );
    return make_tensor(x.data(), im2col_layout);
}

struct GemmConfigSm80 {
   public:
    // 128x128x64 blocks seems to be a good default
    static constexpr int64_t BLK_M = 128;
    static constexpr int64_t BLK_N = 128;
    static constexpr int64_t BLK_K = 64;
    static constexpr int64_t NumThreads = 128;  // 4 warps

   private:
    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<ct::half_t>;
    static constexpr int SmemAtomInner = std::min(64, static_cast<int>(BLK_K));
    static constexpr int SmemAtomOuter = ElemsPerLoad;
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad;

    using BlockShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using BlockShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;

    // The layout of one tile of the smem block, will be tiled to fill the entire block.
    // The choice of this layout is important for performance.
    // Swizzling reduces shared memory bank conflicts.
    using SmemLayoutAtom = decltype(ct::composition(ct::Swizzle<3, 3, 3>{},
                                                    ct::Layout<
                                                        ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                                        ct::Stride<Int<SmemAtomInner>, Int<1>>>{}));

   public:
    // Layout of each block of A/B in shared memory
    using SmemLayoutA = decltype(ct::tile_to_shape(SmemLayoutAtom{}, BlockShapeA{}));
    using SmemLayoutB = decltype(ct::tile_to_shape(SmemLayoutAtom{}, BlockShapeB{}));

   private:
    // The copy atom for gmem -> smem (read A/B) or rmem -> gmem (store C).
    using GmemCopyAtom = ct::Copy_Atom<ct::SM80_CP_ASYNC_CACHEALWAYS<ct::uint128_t>, ct::half_t>;
    // The thread layout for one tile of the gmem -> smem copy.
    using GmemCopyThreadLayoutA = ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                                             ct::Stride<Int<ThreadsPerRow>, Int<1>>>;
    // The value layout for each thread in the gmem -> smem copy.
    using GmemCopyValLayoutA = ct::Layout<ct::Shape<Int<1>, Int<ElemsPerLoad>>>;

   public:
    // Tiled copy of A/B from gmem -> smem
    using GmemCopyA = decltype(ct::make_tiled_copy(GmemCopyAtom{},
                                                   GmemCopyThreadLayoutA{},
                                                   GmemCopyValLayoutA{}));
    using GmemCopyB = GmemCopyA;

   private:
    // The atom of the smem -> rmem copy for A/B. Loads 4 8x8 matrices (distributed across threads) at a time.
    using SmemCopyAtom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, ct::half_t>;
    // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x16 mma (with tensor cores).
    using MmaAtom = ct::MMA_Atom<ct::SM80_16x8x16_F32F16F16F32_TN>;
    // We have 128 threads, so we use 4 warps laid out in 2x2x1.
    using MmaAtomLayout = ct::Layout<ct::Shape<Int<2>, Int<2>, Int<1>>>;
    // We want to use the `ldmatrix.x4.m8n8` instruction which loads 4 8x8 matrices for maximum efficiency.
    // To make the operands A and B divisible into 4 8x8 matrices, we expand the problem size for each warp to 16x16x16.
    // Accounting for the fact that we use 4 warps laid out in 2x2x1, the full tile size is 32x32x16.
    using MmaTiledShape = ct::Tile<Int<32>, Int<32>, Int<16>>;

   public:
    // Tiled mma operation
    using TiledMMA = ct::TiledMMA<MmaAtom, MmaAtomLayout, MmaTiledShape>;
    // Tiled copy of A from smem -> rmem
    using SmemCopyA = decltype(ct::make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    // Tiled copy of B from smem -> rmem
    using SmemCopyB = decltype(ct::make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));
};

using GemmConfig = GemmConfigSm80;

template <typename LayoutImg, typename LayoutKernel, typename LayoutOut>
__global__ void conv2d(
    ct::Tensor<Gmem<ct::half_t>, LayoutImg> img,        // (N H W C)
    ct::Tensor<Gmem<ct::half_t>, LayoutKernel> kernel,  // (K R S C)
    ct::Tensor<Gmem<ct::half_t>, LayoutOut> out         // (N H W K)
) {
    int64_t N = ct::size<0>(img);
    int64_t H = ct::size<1>(img);
    int64_t W = ct::size<2>(img);
    // int64_t C = img.shape(3);
    // int64_t K = kernel.shape(0);
    int64_t R = ct::size<1>(kernel);
    int64_t S = ct::size<2>(kernel);
    // assert(C == kernel.shape(3));

    int64_t block_idx_m = blockIdx.x;
    int64_t block_idx_n = blockIdx.y;

    auto img_grouped = ct::group_modes<0, 3>(img);  // ((N H W) C)
    auto img_tile = ct::make_tile(ct::make_tile(Int<1>{}, Int<1>{}, Int<GemmConfig::BLK_M>{}), Int<GemmConfig::BLK_K>{});
    auto img_blk = ct::local_tile(img_grouped, img_tile, ct::make_coord(block_idx_m, _));  // (BLK_M BLK_K N_BLK_K)

    auto kernel_grouped = ct::group_modes<1, 4>(kernel);  // (K (R S C))
    auto kernel_tile = ct::make_tile(Int<GemmConfig::BLK_N>{}, ct::make_tile(Int<1>{}, Int<1>{}, Int<GemmConfig::BLK_K>{}));
    auto kernel_blk = ct::local_tile(kernel_grouped, kernel_tile, ct::make_coord(block_idx_n, _));  // (BLK_N BLK_K K_BLK_K)

    auto out_grouped = ct::group_modes<0, 3>(out);  // ((N H W) K)
    auto out_tile = ct::make_tile(ct::make_tile(Int<1>{}, Int<1>{}, Int<GemmConfig::BLK_M>{}), Int<GemmConfig::BLK_N>{});
    auto out_blk = ct::local_tile(out_grouped, out_tile, ct::make_coord(block_idx_m, block_idx_n));  // (BLK_M BLK_N)

    // Allocate shared memory for the operands
    typename GemmConfig::SmemLayoutA smem_layout_A;
    typename GemmConfig::SmemLayoutB smem_layout_B;
    __shared__ __align__(sizeof(ct::uint128_t)) ct::half_t sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ __align__(sizeof(ct::uint128_t)) ct::half_t sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

    // if (ct::thread0()) {
    //     ct::print("img\n");
    //     ct::print(img_grouped);
    //     ct::print("\n");
    //     ct::print("img_blk\n");
    //     ct::print(img_blk);
    //     ct::print("\n");
    //     // ct::print("kernel\n");
    //     // ct::print(kernel_grouped);
    //     // ct::print("\n");
    //     ct::print("kernel_blk\n");
    //     ct::print(kernel_blk);
    //     ct::print("\n");
    //     // ct::print("out\n");
    //     // ct::print(out_grouped);
    //     // ct::print("\n");
    //     ct::print("out_blk\n");
    //     ct::print(out_blk);
    //     ct::print("\n");
    // }

    typename GemmConfig::GmemCopyA gmem_copy_A;
    auto thread_copy_A = gmem_copy_A.get_thread_slice(threadIdx.x);
    auto id_A = ct::make_identity_tensor(ct::make_shape(ct::size<0>(img_blk), ct::size<1>(img_blk)));
    auto src_id_A = thread_copy_A.partition_S(id_A);  // (COPY_V COPY_M COPY_K)

    for (int64_t r = -R / 2; r < R - R / 2; ++r) {
        for (int64_t s = -S / 2; s < S - S / 2; ++s) {
            auto pred_A = ct::make_tensor<bool>(ct::select<1, 2>(src_id_A.shape()), ct::Stride<Int<1>, Int<0>>{});  // (COPY_M COPY_K)
            for (int64_t i = 0; i < ct::size<0>(pred_A); ++i) {
                auto global_idx_m = block_idx_m * GemmConfig::BLK_M + ct::get<0>(src_id_A(0, i, 0));
                auto coord_nhw = ct::idx2crd(global_idx_m, ct::shape<0>(img_grouped), ct::compact_row_major(ct::shape<0>(img_grouped)));
                int64_t img_h = ct::get<1>(coord_nhw) + r;
                int64_t img_w = ct::get<2>(coord_nhw) + s;
                pred_A(i, 0) = (0 <= img_h && img_h < H && 0 <= img_w && img_w < W);
                if (ct::thread0()) {
                    // ct::print("shape<0>(img_grouped):\n");
                    // ct::print(ct::shape<0>(img_grouped));
                    // ct::print("\n");
                    // ct::print("stride<0>(img_grouped):\n");
                    // ct::print(ct::stride<0>(img_grouped));
                    // ct::print("\n");
                    ct::print("img_h: %d, img_w: %d, pred_A: %d\n", static_cast<int>(img_h), static_cast<int>(img_w), static_cast<int>(pred_A(i, 0)));
                }
            }
            if (ct::thread0()) {
                ct::print("\n");
            }

            auto N_BLK_K = ct::size<2>(img_blk);
            for (int64_t k = 0; k < N_BLK_K; ++k) {
                auto src_A = thread_copy_A.partition_S(img_blk(_, _, k));  // (COPY_V COPY_M COPY_K)
                auto dst_A = thread_copy_A.partition_D(sA);
                ct::copy_if(gmem_copy_A, pred_A, src_A, dst_A);
                ct::cp_async_wait<0>();
                __syncthreads();
                if (ct::thread0()) {
                    ct::print_tensor(sA);
                    ct::print("\n");
                }
            }
        }
    }

    if (ct::thread0()) {
        ct::print("img_blk\n");
        ct::print(img_blk);
        ct::print("\n");
        // ct::print("src_A\n");
        // ct::print(src_A);
        // ct::print("\n");
        ct::print("src_id_A\n");
        ct::print_tensor(src_id_A);
        ct::print("\n");
    }
}

void set_arange(ct::half_t *data, int64_t size) {
    std::vector<ct::half_t> host_data(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data[i] = static_cast<ct::half_t>(static_cast<float>(i) / 100);
    }
    CUTE_CHECK_ERROR(cudaMemcpy(data, host_data.data(), size * sizeof(ct::half_t), cudaMemcpyHostToDevice));
}

int main(int argc, char const *argv[]) {
    int64_t N = 256;
    int64_t H = 16;
    int64_t W = 16;
    int64_t C = 64;
    int64_t K = 128;
    int64_t R = 3;
    int64_t S = 3;

    ct::half_t *data_img;
    ct::half_t *data_kernel;
    ct::half_t *data_out;
    CUTE_CHECK_ERROR(cudaMalloc(&data_img, N * H * W * C * sizeof(ct::half_t)));
    CUTE_CHECK_ERROR(cudaMalloc(&data_kernel, K * R * S * C * sizeof(ct::half_t)));
    CUTE_CHECK_ERROR(cudaMalloc(&data_out, N * H * W * K * sizeof(ct::half_t)));
    set_arange(data_img, N * H * W * C);
    set_arange(data_kernel, K * R * S * C);
    set_arange(data_out, N * H * W * K);

    auto img = ct::make_tensor(ct::make_gmem_ptr(data_img), ct::make_shape(N, H, W, C), ct::GenRowMajor{});
    auto kernel = ct::make_tensor(ct::make_gmem_ptr(data_kernel), ct::make_shape(K, R, S, C), ct::GenRowMajor{});
    auto out = ct::make_tensor(ct::make_gmem_ptr(data_out), ct::make_shape(N, H, W, K), ct::GenRowMajor{});

    dim3 block_dim((N * H * W / GemmConfigSm80::BLK_M), (K / GemmConfigSm80::BLK_N));
    dim3 thread_dim(GemmConfig::NumThreads);
    conv2d<<<block_dim, thread_dim>>>(img, kernel, out);

    cudaDeviceSynchronize();

    CUTE_CHECK_ERROR(cudaFree(data_img));
    CUTE_CHECK_ERROR(cudaFree(data_kernel));
    CUTE_CHECK_ERROR(cudaFree(data_out));

    return 0;
}
