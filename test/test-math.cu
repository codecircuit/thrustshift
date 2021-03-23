#include <iostream>
#include <random>
#include <algorithm>
#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/math.h>
#include <thrustshift/random.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_math) {
	BOOST_TEST(ceil_divide(10, 10) == 1);
	BOOST_TEST(ceil_divide(10, 11) == 1);
	BOOST_TEST(ceil_divide(10, 9) == 2);
	BOOST_TEST(ceil_divide(10, 5) == 2);
	BOOST_TEST(ceil_divide(10, 4) == 3);
	BOOST_TEST(ceil_divide(0, 0) == 0);
	BOOST_TEST(ceil_divide(0, 10) == 0);
	BOOST_TEST(ceil_divide(-10, -10) == 1);
	BOOST_TEST(ceil_divide(-10, -11) == 1);
	BOOST_TEST(ceil_divide(-10, 3) == -4);
	BOOST_TEST(ceil_divide(-10, 5) == -2);
	BOOST_TEST(ceil_divide(-10, -5) == 2);
	BOOST_TEST(ceil_divide(-10, 6) == -2);
}

BOOST_AUTO_TEST_CASE(test_abs_view) {

	constexpr int N = 6;
	std::vector<double> v(N);
	v[0] = -9.0;
	v[1] = 8.0;
	v[2] = 0.5;
	v[3] = -3.5;
	v[4] = 1.2;
	v[5] = -0.6;

	AbsView<double>* begin = reinterpret_cast<AbsView<double>*>(v.data());
	auto end = begin + N;
	std::sort(begin, end);

	BOOST_TEST(v[0] == 0.5);
	BOOST_TEST(v[1] == -0.6);
	BOOST_TEST(v[2] == 1.2);
	BOOST_TEST(v[3] == -3.5);
	BOOST_TEST(v[4] == 8.0);
	BOOST_TEST(v[5] == -9.0);
}
