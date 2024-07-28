#include <cute/tensor.hpp>

namespace ct = cute;
using ct::Int;

/**
 * logical_product_inverse(select<1, 0>(logical_product(b, a)), b) = a
 * select<1, 0>(logical_product(b, a)) = (b* a, b)
 */
template <typename LayoutAB, typename LayoutB>
auto logical_product_inverse(LayoutAB ab, LayoutB b) {
    auto b_complement_a = ct::layout<0>(ab);
    std::cout << "(b* a)*" << std::endl
              << ct::complement(b_complement_a, ct::cosize(b_complement_a)) << std::endl;
    std::cout << "b*" << std::endl
              << ct::complement(b, ct::cosize(b_complement_a)) << std::endl;
    // auto b = ct::layout<1>(ab);
    auto b_complement = ct::complement(b, ct::cosize(b_complement_a));
    auto b_complement_inv = ct::left_inverse(b_complement);
    auto a = ct::composition(b_complement_inv, b_complement_a);
    return a;
}

template <int I, typename Layout>
auto remove_mode(const Layout &x) {
    // auto to_remove = ct::layout<I>(x);
    auto removed_cosize = ct::cosize(ct::layout<I>(x));
    auto new_layout = ct::make_layout(
        ct::remove<I>(ct::shape(x)),
        ct::remove<I>(ct::stride(x)));

    ct::transform_layout

        return new_layout;
}

int main(int argc, char const *argv[]) {
    auto a = ct::make_layout(ct::make_shape(Int<2>{}, Int<3>{}), ct::GenRowMajor{});
    std::cout << "a" << std::endl
              << a << std::endl;
    auto b = ct::make_layout(ct::make_shape(Int<4>{}), ct::GenRowMajor{});
    std::cout << "b" << std::endl
              << b << std::endl;

    auto ab = ct::select<1, 0>(ct::logical_product(b, a));
    // auto ab = ct::make_layout(ct::make_shape(ct::make_shape(Int<2>{}, Int<3>{}), Int<4>{}), ct::GenRowMajor{});
    std::cout << "ab" << std::endl
              << ab << std::endl;
    // auto a = ct::make_layout(ct::make_shape(Int<4>{}));
    // std::cout << "a" << std::endl
    //           << a << std::endl;

    // auto a_ = logical_product_inverse(ab, b);
    // std::cout << "a_" << std::endl
    //           << a_ << std::endl;

    std::cout << "remove 1" << std::endl
              << remove_mode<1>(ab) << std::endl;

    return 0;
}
