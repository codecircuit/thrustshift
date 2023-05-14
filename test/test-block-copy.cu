#include <algorithm>
#include <variant>
#include <vector>

#include <Eigen/Sparse>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/copy.h>
#include <thrustshift/managed-vector.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

template <int BLOCK_DIM, int NUM_ELEMENTS>
__global__ void do_block_copy_simple(gsl_lite::span<const int> src,
                                     gsl_lite::span<int> dst) {
	block_copy<BLOCK_DIM, NUM_ELEMENTS>(src.begin(), dst.begin());
}

template <int BLOCK_DIM, int N>
void do_simple_block_copy_test() {
	auto device = cuda::device::current::get();
	managed_vector<int> src(N);
	std::iota(src.begin(), src.end(), 0);
	managed_vector<int> dst(N);

	cuda::launch(do_block_copy_simple<BLOCK_DIM, N>,
	             cuda::make_launch_config(1, BLOCK_DIM),
	             gsl::make_span(src),
	             gsl::make_span(dst));

	device.synchronize();
	BOOST_TEST(std::equal(src.begin(), src.end(), dst.begin()));
}

BOOST_AUTO_TEST_CASE(test_block_copy_simple) {

	auto device = cuda::device::current::get();
	{
		constexpr int N = 100;
		constexpr int BLOCK_DIM = 128;
		do_simple_block_copy_test<BLOCK_DIM, N>();
	}
	{
		constexpr int N = 1;
		constexpr int BLOCK_DIM = 128;
		do_simple_block_copy_test<BLOCK_DIM, N>();
	}
	{
		constexpr int N = 165;
		constexpr int BLOCK_DIM = 1;
		do_simple_block_copy_test<BLOCK_DIM, N>();
	}
	{
		constexpr int N = 1655;
		constexpr int BLOCK_DIM = 133;
		do_simple_block_copy_test<BLOCK_DIM, N>();
	}
}
