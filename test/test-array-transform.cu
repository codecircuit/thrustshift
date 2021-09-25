#include <algorithm>
#include <array>
#include <gsl-lite/gsl-lite.hpp>
#include <iostream>
#include <random>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/transform.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_array_transform) {

	std::array<int, 4> arr({1, 2, 3, 4});

	auto result = array::transform(arr, [](int i) { return i * 2; });
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(result[i] == (i + 1) * 2);
	}
}
