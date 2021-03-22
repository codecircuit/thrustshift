#include <algorithm>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <matrixmarket/eigen.hpp>

#include <thrustshift/memory-resource.h>
#include <thrustshift/not-a-vector.h>

namespace bdata = boost::unit_test::data;
namespace utf = boost::unit_test;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_not_a_vector) {
	pmr::delayed_pool_type<pmr::managed_resource_type> mres;
	constexpr int N = 512;
	auto [nav, s] = make_not_a_vector_and_span<float>(N, mres);
	BOOST_TEST(nav.to_span().data() == s.data());
	static_assert(std::is_same<decltype(s), gsl_lite::span<float>>::value);
}
