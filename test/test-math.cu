#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/math.h>

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
	BOOST_CHECK_THROW(ceil_divide(10, 0), std::exception);
	BOOST_TEST(ceil_divide(-10, 6) == -2);
}
