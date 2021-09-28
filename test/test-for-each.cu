#include <algorithm>
#include <gsl-lite/gsl-lite.hpp>
#include <iostream>
#include <random>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrust/tuple.h>

#include <thrustshift/for-each.h>
#include <thrustshift/functional.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_for_each) {
	std::tuple<double, double, double> tup(1, 2, 3);
	std::tuple<double, double, double> tup2(1, 2, 3);

	tuple::for_each(tup, [](auto& d) { d += 1; });

	BOOST_TEST(std::get<0>(tup) == 2);
	BOOST_TEST(std::get<1>(tup) == 3);
	BOOST_TEST(std::get<2>(tup) == 4);

	tuple::for_each(tup, tup2, [](auto& a, auto& b) { a = a + b; });

	BOOST_TEST(std::get<0>(tup) == 3);
	BOOST_TEST(std::get<1>(tup) == 5);
	BOOST_TEST(std::get<2>(tup) == 7);

	tuple::for_each(tup, plus_equal_assign_constant<double&, double>(1.0));

	BOOST_TEST(std::get<0>(tup) == 4);
	BOOST_TEST(std::get<1>(tup) == 6);
	BOOST_TEST(std::get<2>(tup) == 8);

	double a = 1;
	double b = 2;
	double c = 3;
	thrust::tuple<double&, double&, double&> tup3(a, b, c);

	tuple::for_each(tup3, plus_equal_assign_constant<double&, double>(1.0));
	BOOST_TEST(a == 2.0);
	BOOST_TEST(b == 3.0);
	BOOST_TEST(c == 4.0);

	thrust::tuple<double, double, double> tup4(1, 2, 3);

	tuple::for_each(tup4, plus_equal_assign_constant<double&, double>(1.0));
	BOOST_TEST(thrust::get<0>(tup4) == 2.0);
	BOOST_TEST(thrust::get<1>(tup4) == 3.0);
	BOOST_TEST(thrust::get<2>(tup4) == 4.0);

	auto tup5 = thrust::make_tuple(1.0, 2.0, 3.0);

	tuple::for_each(tup5, plus_equal_assign_constant<double&, double>(1.0));
	BOOST_TEST(thrust::get<0>(tup5) == 2.0);
	BOOST_TEST(thrust::get<1>(tup5) == 3.0);
	BOOST_TEST(thrust::get<2>(tup5) == 4.0);
}

BOOST_AUTO_TEST_CASE(test_for_each_with_thrust_tuple) {
	thrust::tuple<double, double, double> tup(1, 2, 3);
	thrust::tuple<double, double, double> tup2(1, 2, 3);

	tuple::for_each(tup, [](auto& d) { d += 1; });

	BOOST_TEST(thrust::get<0>(tup) == 2);
	BOOST_TEST(thrust::get<1>(tup) == 3);
	BOOST_TEST(thrust::get<2>(tup) == 4);

	tuple::for_each(tup, tup2, [](auto& a, auto& b) { a = a + b; });

	BOOST_TEST(thrust::get<0>(tup) == 3);
	BOOST_TEST(thrust::get<1>(tup) == 5);
	BOOST_TEST(thrust::get<2>(tup) == 7);
}
