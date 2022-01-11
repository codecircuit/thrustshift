#include <algorithm>
#include <array>
#include <gsl-lite/gsl-lite.hpp>
#include <iostream>
#include <random>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/types-io.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_tuple_to_string) {

	std::tuple<float, double, int> tup{1, 2, 3};

	auto str = tuple::to_string(tup);
	BOOST_TEST(str == "1,2,3");
}
