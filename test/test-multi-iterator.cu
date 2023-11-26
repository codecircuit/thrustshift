#include <algorithm>
#include <iostream>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/arithmetic-multi-iterator-reference.h>
#include <thrustshift/multi-iterator.h>

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

using namespace thrustshift;

BOOST_AUTO_TEST_CASE(test_multi_iterator) {

	std::vector<double> v(30);
	double* it0 = v.data();
	double* it1 = v.data() + 10;
	double* it2 = v.data() + 20;
	multi_iterator<double*, 3> mi({it0, it1, it2});
	mi.size();

	thrust::tuple<double, double, double> tup_values(1.1, 1.2, 1.3);
	thrust::tuple<double, double, double> tup_values1(2.1, 2.2, 2.3);
	auto tup = *mi;
	tup = tup_values;
	mi[2] = tup_values1;

	BOOST_TEST(v[0] == 1.1);
	BOOST_TEST(v[10] == 1.2);
	BOOST_TEST(v[20] == 1.3);
	BOOST_TEST(v[2] == 2.1);
	BOOST_TEST(v[12] == 2.2);
	BOOST_TEST(v[22] == 2.3);
}

BOOST_AUTO_TEST_CASE(test_arithmetic_multi_iterator) {

	std::vector<double> v(30);
	double* it0 = v.data();
	double* it1 = v.data() + 10;
	double* it2 = v.data() + 20;
	using Ref = arithmetic_multi_iterator_reference<double&, 3>;
	multi_iterator<double*, 3, Ref> mi({it0, it1, it2});
	mi.size();

	thrust::tuple<double, double, double> tup_values(1.1, 1.2, 1.3);
	thrust::tuple<double, double, double> tup_values1(2.1, 2.2, 2.3);
	auto tup = *mi;

	static_assert(is_tuple<decltype(tup_values)>::value);
	static_assert(!is_tuple<std::vector<double>>::value);

	tup = tup_values;
	mi[2] = tup_values1;

	arithmetic_tuple<double, 3> atup(tup);

	atup += 3.0;
	BOOST_TEST(thrust::get<0>(atup) == 4.1, tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<1>(atup) == 4.2, tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<2>(atup) == 4.3, tt::tolerance(0.0001));

	atup += tup;

	BOOST_TEST(thrust::get<0>(atup) == 5.2, tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<1>(atup) == 5.4, tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<2>(atup) == 5.6, tt::tolerance(0.0001));

	auto add_result = atup + tup;

	BOOST_TEST(thrust::get<0>(add_result) == 6.3, tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<1>(add_result) == 6.6, tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<2>(add_result) == 6.9, tt::tolerance(0.0001));

	auto add_result2 = add_result + add_result;

	BOOST_TEST(thrust::get<0>(add_result2) == (2 * 6.3), tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<1>(add_result2) == (2 * 6.6), tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<2>(add_result2) == (2 * 6.9), tt::tolerance(0.0001));

	auto other_result = add_result + 7;

	BOOST_TEST(thrust::get<0>(other_result) == (6.3 + 7),
	           tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<1>(other_result) == (6.6 + 7),
	           tt::tolerance(0.0001));
	BOOST_TEST(thrust::get<2>(other_result) == (6.9 + 7),
	           tt::tolerance(0.0001));

	BOOST_TEST(v[0] == 1.1);
	BOOST_TEST(v[10] == 1.2);
	BOOST_TEST(v[20] == 1.3);
	BOOST_TEST(v[2] == 2.1);
	BOOST_TEST(v[12] == 2.2);
	BOOST_TEST(v[22] == 2.3);

	for (int i = 0; i < 10; ++i) {
		mi[i] += 3 * i;
	}

	mi[7] = (mi[6] + mi[5] * 3.1654) / mi[0];

	std::array<double, 3> arr{};
	for (int j = 0; j < 3; ++j) {
		arr[j] = (v[j * 10 + 6] + v[j * 10 + 5] * 3.1654) / v[j * 10];
	}
	BOOST_TEST(arr[0] == v[7]);
	BOOST_TEST(arr[1] == v[17]);
	BOOST_TEST(arr[2] == v[27]);

	auto prev = thrust::get<0>(mi[5]);
	mi[3] = 5 * mi[5];
	BOOST_TEST(prev * 5 == thrust::get<0>(mi[3]));
}
