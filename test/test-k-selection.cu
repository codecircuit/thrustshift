#include <algorithm>
#include <bitset>
#include <numeric>
#include <type_traits>
#include <variant>
#include <vector>

#include <Eigen/Sparse>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/CSR.h>
#include <thrustshift/k-selection.h>
#include <thrustshift/managed-vector.h>
#include <thrustshift/memory-resource.h>
#include <thrustshift/random.h>
#include <thrustshift/sort.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

template <typename Range>
auto k_largest_abs_values_gold(Range&& r, int k) {

	gsl_Expects(k >= 1);
	using T = typename std::remove_reference<Range>::type::value_type;
	const int N = r.size();
	std::vector<std::tuple<T, int>> v(N);
	for (int i = 0; i < N; ++i) {
		v[i] = std::tuple<T, int>{r[i], i};
	}
	std::sort(v.begin(), v.end(), [](auto a, auto b){ return std::abs(std::get<0>(a)) > std::abs(std::get<0>(b)); });
	v.resize(k);
	return v;
}



BOOST_AUTO_TEST_CASE(test_k_selection) {

	constexpr std::size_t N = 100;

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	thrustshift::managed_vector<int> v(N);
	thrustshift::managed_vector<int> histogram(256);

	std::default_random_engine rng;
	std::uniform_int_distribution<int> dist(0, 765);
	for (auto& e : v) {
		e = dist(rng);
	}
	// for (int i = 0; i < 16; ++i) {
	// 	v[i] = 2;
	// }
	// v[N/2] = std::numeric_limits<int>::max();
	// v[N/2+1] = std::numeric_limits<int>::max()/2;
	for (int i = 0; i < N; ++i) {
		std::cout << "v[" << i << "] = " << v[i] << std::endl;
	}

	constexpr int k = 12;
	const auto kl = k_largest_abs_values_gold(v, k);
	std::cout << "k largest values from cpu:\n";
	for (auto& e : kl) {
		std::cout << "(" << std::get<0>(e) << ", " << std::get<1>(e) << ")" << std::endl;
	}

	thrustshift::pmr::delayed_pool_type<thrustshift::pmr::managed_resource_type>
	    delayed_memory_resource;

	auto unary_functor = [] __device__(int x) { return std::abs(x); };

	async::bin_values256<int>(
	    stream, v, histogram, 0, 0, unary_functor, delayed_memory_resource);
	device.synchronize();

	auto [prefix, bit_offset] = thrustshift::k_largest_values_abs_radix<int>(
	    stream, v, k, delayed_memory_resource);
	std::cout << "prefix  = " << std::bitset<64>(prefix) << std::endl;
	std::cout << "bit_offset = " << bit_offset << std::endl;

	for (int i = 0; i < 256; ++i) {
		std::cout << "histogram[" << i << "] = " << histogram[i] << std::endl;
	}

	thrustshift::managed_vector<thrust::tuple<int, int>> selected_values(N);

	select_k_largest_values_abs<int>(
	    stream, v, selected_values, k, delayed_memory_resource);
	for (int i = 0; i < N; ++i) {
		std::cout << "selected_values[" << i << "] = ("
		          << thrust::get<0>(selected_values[i]) << ", "
		          << thrust::get<1>(selected_values[i]) << ")" << std::endl;
	}
}
