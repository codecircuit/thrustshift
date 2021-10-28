#include <type_traits>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrust/tuple.h>

#include <cuda/define_specifiers.hpp>
#include <thrustshift/type-traits.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

namespace {

template<class T>
void f(T&&) {
	static_assert(is_tuple<T>::value);
}

}

BOOST_AUTO_TEST_CASE(test_type_traits) {

	auto seq = std::make_index_sequence<5>();
	const auto len = get_integer_sequence_length(seq);
	BOOST_TEST(len == 5);

	using T0 = thrust::tuple<double, double>;
	using T1 = const thrust::tuple<double, double>&;
	using T2 = thrust::tuple<double, double>&&;
	static_assert(is_tuple<T0>::value);
	static_assert(is_tuple<T1>::value);
	static_assert(is_tuple<T2>::value);

	T0 a;
	T1 b(a);
	f(a);
	f(b);

	using T3 = make_uintegral_of_equal_size<double>::type;
	using T4 = make_uintegral_of_equal_size<float>::type;
	using T5 = make_uintegral_of_equal_size<int>::type;
}

