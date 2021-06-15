#include <algorithm>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/memory-resource.h>
#include <thrustshift/not-a-vector.h>

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
	{
		auto [nav, s] = make_not_a_vector_and_span<int>(N, mres);
	}
	{
		// Here the float buffer might be reused if the alignment is the same
		auto [nav, s] = make_not_a_vector_and_span<bool>(N * sizeof(float), mres);
	}
	{
		// Here the float buffer might be reused if the alignment is the same
		auto [nav, s] = make_not_a_vector_and_span<double>(N / sizeof(float) * sizeof(double), mres);
	}
	{
		auto [nav, s] = make_not_a_vector_and_span<float>(N, mres);
		BOOST_TEST(ptr0 == s.data());
	}
}
