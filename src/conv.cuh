#pragma once

#include <cute/tensor.hpp>

namespace ct = cute;
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

#define DEBUG_THREAD ct::thread(0, 1)

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
    // using SmemLayoutAtom = decltype(ct::composition(ct::Swizzle<3, 3, 3>{},
    //                                                 ct::Layout<
    //                                                     ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
    //                                                     ct::Stride<Int<SmemAtomInner>, Int<1>>>{}));
    using SmemLayoutAtom = ct::Layout<
        ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
        ct::Stride<Int<SmemAtomInner>, Int<1>>>;

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

template <typename GemmConfig, typename LayoutBlkC>
struct SmemGemm {
   private:
    ct::Tensor<Gmem<ct::half_t>, LayoutBlkC> &C;
    typename GemmConfig::TiledMMA tiled_mma;
    typename GemmConfig::SmemCopyA smem_tiled_copy_A;
    typename GemmConfig::SmemCopyB smem_tiled_copy_B;

    decltype(tiled_mma.get_thread_slice(0u)) thread_mma;
    decltype(thread_mma.partition_fragment_C(C)) C_frag;

   public:
    CUTE_DEVICE SmemGemm(ct::Tensor<Gmem<ct::half_t>, LayoutBlkC> &C_)
        : C(C_),
          thread_mma(tiled_mma.get_thread_slice(threadIdx.x)),
          C_frag(thread_mma.partition_fragment_C(C)) {
        ct::clear(C_frag);
    }

    // Perform Smem GEMM: C += A @ B
    CUTE_DEVICE void operator()(
        const ct::Tensor<Smem<ct::half_t>, typename GemmConfig::SmemLayoutA> &sA,
        const ct::Tensor<Smem<ct::half_t>, typename GemmConfig::SmemLayoutB> &sB) {
        // Allocate registers distributed across threads to store operands
        auto A_frag = thread_mma.partition_fragment_A(sA);
        auto B_frag = thread_mma.partition_fragment_B(sB);

        // Load A and B from smem to registers (distributed across threads)
        auto thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
        auto sA_to_rA_src = thr_copy_A.partition_S(sA);   // COPY_V, COPY_M, COPY_K
        auto sA_to_rA_dst = thr_copy_A.retile_D(A_frag);  // COPY_V, COPY_M, COPY_K
        ct::copy(smem_tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);

        auto thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
        auto sB_to_rB_src = thr_copy_B.partition_S(sB);   // COPY_V, COPY_N, COPY_K
        auto sB_to_rB_dst = thr_copy_B.retile_D(B_frag);  // COPY_V, COPY_N, COPY_K
        ct::copy(smem_tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);

        // Perform GEMM
        ct::gemm(tiled_mma, A_frag, B_frag, C_frag);
    }

    // Write back result to gmem
    CUTE_DEVICE void write_back() {
        // if (DEBUG_THREAD) {
        //     ct::print_tensor(C_frag);
        //     ct::print("\n");
        // }
        auto C_frag_out = thread_mma.partition_C(C);  // Corresponding location in output tensor
        ct::copy(C_frag, C_frag_out);
    }
};

// Reordering the block access pattern helps to improve L2 cache hit rate.
// Triton's doc for matmul has a nice explanation: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
// For m = 3, n = 4, group_size_m = 2, produces the coordiantes in the following order:
//  |  1 |  3 |  5 |  7 |
//  |  2 |  4 |  6 |  8 |
//  |  9 | 10 | 11 | 12 |
CUTE_DEVICE std::tuple<int, int> threadblock_swizzle(int group_size_m) {
    // Assume a 2D grid of thread blocks of (m=gridDim.x, n=gridDim.y)
    // Note that the lauch order of thread blocks is column-major. i.e. x varies faster than y.
    int idx = blockIdx.y * gridDim.x + blockIdx.x;  // 1D index of the thread block
    int blocks_per_group = group_size_m * gridDim.y;
    int first_block_idx_m = (idx / blocks_per_group) * group_size_m;
    group_size_m = min(gridDim.x - first_block_idx_m, group_size_m);  // Min to handle edge case of m % group_size_m != 0
    int block_idx_m = first_block_idx_m + (idx % group_size_m);
    int block_idx_n = (idx % blocks_per_group) / group_size_m;
    return std::make_tuple(block_idx_m, block_idx_n);
}

template <typename T, typename SrcLayout, typename DstLayout, typename TiledCopy>
CUTE_DEVICE void load_block_from_gmem_to_smem(
    const ct::Tensor<Gmem<T>, SrcLayout> &src,
    const ct::Tensor<Smem<T>, DstLayout> &dst,
    TiledCopy tiled_copy) {
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_frag = thread_copy.partition_S(src);
    auto dst_frag = thread_copy.partition_D(dst);
    ct::copy(tiled_copy, src_frag, dst_frag);
}

template <typename GemmConfig, typename LayoutInput, typename LayoutKernel, typename LayoutOut>
__global__ void conv2d_kernel(
    ct::Tensor<Gmem<ct::half_t>, LayoutInput> img,      // (N H W C)
    ct::Tensor<Gmem<ct::half_t>, LayoutKernel> kernel,  // (K R S C)
    ct::Tensor<Gmem<ct::half_t>, LayoutOut> out,        // (N P=H Q=W K)
    int64_t group_size_m) {
    int64_t N = ct::size<0>(img);
    int64_t H = ct::size<1>(img);
    int64_t W = ct::size<2>(img);
    int64_t C = ct::size<3>(img);
    int64_t K = ct::size<0>(kernel);
    int64_t R = ct::size<1>(kernel);
    int64_t S = ct::size<2>(kernel);

    auto [block_idx_m, block_idx_n] = threadblock_swizzle(group_size_m);

    // if (block_idx_m == 0) {
    //     return;
    // }

    // auto kernel_grouped = ct::group_modes<1, 4>(kernel);  // (K (R S C))
    auto kernel_tile = ct::make_tile(Int<GemmConfig::BLK_N>{}, Int<1>{}, Int<1>{}, Int<GemmConfig::BLK_K>{});
    auto kernel_blk = ct::local_tile(kernel, kernel_tile, ct::make_coord(block_idx_n, _, _, _));  // (BLK_N 1 1 BLK_K R S K_BLK_K)

    // auto out_grouped = ct::group_modes<0, 3>(out);  // ((N H W) K)
    auto out_grouped = ct::make_tensor(out.data(), ct::make_shape(N * H * W, K), ct::GenRowMajor{});
    auto out_tile = ct::make_shape(Int<GemmConfig::BLK_M>{}, Int<GemmConfig::BLK_N>{});
    auto out_blk = ct::local_tile(out_grouped, out_tile, ct::make_coord(block_idx_m, block_idx_n));  // (BLK_M BLK_N)
    // if (DEBUG_THREAD) {
    //     ct::print("out_grouped\n");
    //     ct::print(out_grouped);
    //     ct::print("\n");
    //     ct::print("out_blk\n");
    //     ct::print(out_blk);
    //     ct::print("\n");
    // }

    // Allocate shared memory for the operands
    typename GemmConfig::SmemLayoutA smem_layout_A;
    typename GemmConfig::SmemLayoutB smem_layout_B;
    __shared__ __align__(sizeof(ct::uint128_t)) ct::half_t sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ __align__(sizeof(ct::uint128_t)) ct::half_t sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

    SmemGemm<GemmConfig, std::decay_t<decltype(out_blk.layout())>> smem_gemm(out_blk);

    typename GemmConfig::GmemCopyA gmem_copy_A;
    auto thread_copy_A = gmem_copy_A.get_thread_slice(threadIdx.x);
    auto id_A = ct::make_identity_tensor(ct::make_shape(Int<GemmConfig::BLK_M>{}, Int<GemmConfig::BLK_K>{}));
    auto src_id_A = thread_copy_A.partition_S(id_A);  // (COPY_V COPY_M COPY_K)

    for (int64_t r = -R / 2; r < R - R / 2; ++r) {
        for (int64_t s = -S / 2; s < S - S / 2; ++s) {
            auto pred_A = ct::make_tensor<bool>(ct::select<1, 2>(src_id_A.shape()), ct::Stride<Int<1>, Int<0>>{});  // (COPY_M COPY_K)
            for (int64_t i = 0; i < ct::size<0>(pred_A); ++i) {
                auto global_idx_m = C * (block_idx_m * GemmConfig::BLK_M + ct::get<0>(src_id_A(0, i, 0)));
                auto coord_nhw = ct::idx2crd(global_idx_m, ct::select<0, 1, 2>(img.shape()), ct::select<0, 1, 2>(img.stride()));
                int64_t img_h = ct::get<1>(coord_nhw) + r;
                int64_t img_w = ct::get<2>(coord_nhw) + s;
                pred_A(i, 0) = (0 <= img_h && img_h < H && 0 <= img_w && img_w < W);
                // if (DEBUG_THREAD) {
                // ct::print("shape<0>(img_grouped):\n");
                // ct::print(ct::shape<0>(img_grouped));
                // ct::print("\n");
                // ct::print("stride<0>(img_grouped):\n");
                // ct::print(ct::stride<0>(img_grouped));
                // ct::print("\n");
                // ct::print("img_h: %d, img_w: %d, pred_A: %d\n", static_cast<int>(img_h), static_cast<int>(img_w), static_cast<int>(pred_A(i, 0)));
                // }
            }
            // if (DEBUG_THREAD) {
            //     ct::print("\n");
            // }

            for (int64_t k = 0; k < C / GemmConfig::BLK_K; ++k) {
                // auto img_grouped = ct::group_modes<0, 3>(ct::domain_offset(ct::make_coord(0, r, s, 0), img));  // ((N H W) C)
                auto img_offset = ct::domain_offset(ct::make_coord(0, r, s, 0), img);
                auto img_grouped = ct::make_tensor(img_offset.data(), ct::make_shape(N * H * W, C), ct::GenRowMajor{});
                auto img_tile = ct::make_tile(Int<GemmConfig::BLK_M>{}, Int<GemmConfig::BLK_K>{});
                auto img_blk = ct::local_tile(img_grouped, img_tile, ct::make_coord(block_idx_m, _));  // (BLK_M BLK_K N_BLK_K)
                auto src_A = thread_copy_A.partition_S(img_blk(_, _, k));                              // (COPY_V COPY_M COPY_K)
                auto dst_A = thread_copy_A.partition_D(sA);
                // if (DEBUG_THREAD) {
                //     ct::print("img\n");
                //     ct::print(img);
                //     ct::print("\n");
                //     ct::print("img_grouped\n");
                //     ct::print(img_grouped);
                //     ct::print("\n");
                //     ct::print("img_blk\n");
                //     ct::print(img_blk);
                //     ct::print("\n");
                //     ct::print("src_A\n");
                //     ct::print(src_A);
                //     ct::print("\n");
                //     ct::print("dst_A\n");
                //     ct::print(dst_A);
                //     ct::print("\n");
                // ct::print("pred_A\n");
                // ct::print(pred_A);
                // ct::print("\n");
                //     ct::print("r: %d, s: %d, k: %d\n", static_cast<int>(r), static_cast<int>(s), static_cast<int>(k));
                // }
                ct::copy_if(gmem_copy_A, pred_A, src_A, dst_A);
                load_block_from_gmem_to_smem(kernel_blk(_, 0, 0, _, r + R / 2, s + S / 2, k), sB, typename GemmConfig::GmemCopyB{});
                ct::cp_async_wait<0>();
                __syncthreads();
                smem_gemm(sA, sB);
                // if (DEBUG_THREAD) {
                //     ct::print_tensor(sA);
                //     ct::print("\n");
                // }
                __syncthreads();
            }
        }
    }

    smem_gemm.write_back();
    ct::cp_async_wait<0>();
    __syncthreads();
}

template <typename LayoutInput, typename LayoutKernel, typename LayoutOutput>
void conv2d(
    const ct::Tensor<Gmem<ct::half_t>, LayoutInput> &input,
    const ct::Tensor<Gmem<ct::half_t>, LayoutKernel> &kernel,
    const ct::Tensor<Gmem<ct::half_t>, LayoutOutput> &output) {
    int64_t N = ct::size<0>(input);
    int64_t H = ct::size<1>(input);
    int64_t W = ct::size<2>(input);
    int64_t C = ct::size<3>(input);
    int64_t K = ct::size<0>(kernel);
    int64_t R = ct::size<1>(kernel);
    int64_t S = ct::size<2>(kernel);

    assert((N * H * W) % GemmConfig::BLK_M == 0 && "N * H * W must be divisible by BLK_M");
    assert(K % GemmConfig::BLK_N == 0 && "K must be divisible by BLK_N");

    dim3 block_dim((N * H * W) / GemmConfig::BLK_M, K / GemmConfig::BLK_N);
    dim3 thread_dim(GemmConfig::NumThreads);
    int num_sms;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    int64_t group_size_m = std::sqrt(num_sms);
    conv2d_kernel<GemmConfig><<<block_dim, thread_dim>>>(input, kernel, output, group_size_m);
}
