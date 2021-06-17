#include <algorithm>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <thrust/reduce.h>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/copy.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>
#include <thrustshift/not-a-vector.h>

#include "./memory-resource-check.h"

namespace bdata = boost::unit_test::data;
namespace utf = boost::unit_test;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_memory_resource) {
	pmr::delayed_pool_type<pmr::managed_resource_type> mres;
	constexpr int N = 512;
	float* ptr0;
	{
		auto [nav, s] = make_not_a_vector_and_span<float>(N, mres);
		ptr0 = s.data();
	}
	{ auto [nav, s] = make_not_a_vector_and_span<int>(N, mres); }
	{
		// Here the float buffer might be reused if the alignment is the same
		auto [nav, s] =
		    make_not_a_vector_and_span<bool>(N * sizeof(float), mres);
	}
	{
		// Here the float buffer might be reused if the alignment is the same
		auto [nav, s] = make_not_a_vector_and_span<double>(
		    N / sizeof(float) * sizeof(double), mres);
	}
	{
		auto [nav, s] = make_not_a_vector_and_span<float>(N, mres);
		BOOST_TEST(ptr0 == s.data());
	}
}

BOOST_AUTO_TEST_CASE(test_memory_resource_with_thrusts_reduce_by_key) {
	pmr::delayed_pool_type<pmr::managed_resource_type> memory_resource;
	auto device = cuda::device::current::get();
	auto stream = device.default_stream();
	{
		const int N = 2401;

		managed_vector<int> keys(N);
		managed_vector<float> values(N);
		managed_vector<int> keys_out(N);
		managed_vector<float> values_out(N);

		std::pmr::polymorphic_allocator<char> alloc(&memory_resource);
		thrust::reduce_by_key(
		    thrust::cuda::par(alloc),
		    keys.data(),
		    keys.data() + keys.size(),
		    values.data(),
		    keys_out.data(),
		    values_out.data(),
		    thrust::equal_to<int>(),
		    [] __device__(float a, float b) { return a + b; });
	}
	touch_all_memory_resource_pages(memory_resource);
}

namespace {

template<class MemoryResource>
std::pmr::vector<float> construct(MemoryResource& mres) {
	const int N = 2401;
	std::pmr::vector<float> v0(2 * N, &mres);
	std::pmr::vector<float> v(N, &mres);
	return v;
}

}

// Closed Issue #2
BOOST_AUTO_TEST_CASE(test_memory_resource_with_pmr_vector_construction, *boost::unit_test::disabled()) {
	pmr::delayed_pool_type<pmr::managed_resource_type> memory_resource;
	{
		auto v = construct(memory_resource);
	}
	touch_all_memory_resource_pages(memory_resource);
}
