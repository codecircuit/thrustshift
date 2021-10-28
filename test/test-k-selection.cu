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

namespace {

template <typename Range>
auto k_largest_abs_values_gold(Range&& r, int k) {

	gsl_Expects(k >= 1);
	using T = typename std::remove_reference<Range>::type::value_type;
	const int N = r.size();
	std::vector<std::tuple<T, int>> v(N);
	for (int i = 0; i < N; ++i) {
		v[i] = std::tuple<T, int>{r[i], i};
	}
	std::sort(v.begin(), v.end(), [](auto a, auto b) {
		return std::abs(std::get<0>(a)) > std::abs(std::get<0>(b));
	});
	v.resize(k);
	return v;
}

template <typename T>
struct k_selection_test_data_t {
	std::vector<T> values;
	int k;
};

template <typename T>
std::ostream& operator<<(std::ostream& os,
                         const k_selection_test_data_t<T>& td) {
	os << "k = " << td.k << '\n';
	const size_t N = td.values.size();
	os << "values = [";
	if (N <= 100 && N > 0) {
		for (size_t i = 0; i < N - 1; ++i) {
			os << td.values[i] << ", ";
		}
		os << td.values[N - 1] << "]\n";
	}
	return os;
}

template<typename T>
struct uniform_distribution_type_proxy {
	using type = std::uniform_int_distribution<T>;
};
template<>
struct uniform_distribution_type_proxy<float> {
	using type = std::uniform_real_distribution<float>;
};
template<>
struct uniform_distribution_type_proxy<double> {
	using type = std::uniform_real_distribution<double>;
};

template <typename T>
auto gen_uniform_values(std::size_t N, T min, T max) {
	std::vector<T> v(N);
	std::default_random_engine rng;
	typename uniform_distribution_type_proxy<T>::type dist(min, max);
	// auto dist = [&] { // NOTE: produces still warnings. Wait for better compiler
	// 	if constexpr (std::is_integral<T>::value) {
	// 		return std::uniform_int_distribution<T>(min, max);
	// 	}
	// 	else {
	// 		return std::uniform_real_distribution<T>(min, max);
	// 	}
	// }();
	for (auto& e : v) {
		e = dist(rng);
	}
	return v;
}

std::vector<k_selection_test_data_t<int>> int_test_datas = {
    {gen_uniform_values<int>(100, 0, 765), 7},
    {gen_uniform_values<int>(100, 0, 765), 1},
    {gen_uniform_values<int>(100, 0, 765), 100},
    {gen_uniform_values<int>(100, 0, std::numeric_limits<int>::max()), 12},
    {gen_uniform_values<int>(100,
                                 std::numeric_limits<int>::min(),
                                 std::numeric_limits<int>::max()),
     13},
    {gen_uniform_values<int>(100, -6478, 765), 56},
    {gen_uniform_values<int>(476168, 0, 765), 10},
    {gen_uniform_values<int>(476168, -6884, 765), 5},
};

std::vector<k_selection_test_data_t<float>> float_test_datas = {
    {gen_uniform_values<float>(100, 0, 765), 7},
    {gen_uniform_values<float>(100, 0, 765), 1},
    {gen_uniform_values<float>(100, 0, 765), 100},
    {gen_uniform_values<float>(100, 0, std::numeric_limits<float>::max()), 12},
    {gen_uniform_values<float>(100,
                                 std::numeric_limits<float>::min(),
                                 std::numeric_limits<float>::max()),
     13},
    {gen_uniform_values<float>(100, -6478, 765), 56},
    {gen_uniform_values<float>(476168, 0, 765), 10},
    {gen_uniform_values<float>(476168, -6884, 765), 5},
};

} // namespace

template <typename T>
void do_k_selection_test(const k_selection_test_data_t<T>& td) {
	const thrustshift::managed_vector<T> v(td.values.begin(), td.values.end());
	const int k = td.k;
	const std::size_t N = v.size();

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	const auto selected_values_gold = k_largest_abs_values_gold(v, k);

	thrustshift::pmr::delayed_pool_type<thrustshift::pmr::managed_resource_type>
	    delayed_memory_resource;

	thrustshift::managed_vector<thrust::tuple<T, int>> selected_values(N);

	select_k_largest_values_abs<T>(
	    stream, v, selected_values, k, delayed_memory_resource);

	std::set<T> gold_selected_unique;
	std::set<T> contender_selected_unique;
	for (const auto& e : selected_values_gold) {
		gold_selected_unique.insert(std::get<0>(e));
	}
	for (const auto& e : selected_values) {
		contender_selected_unique.insert(thrust::get<0>(e));
	}
	const bool p = gold_selected_unique == contender_selected_unique;
	for (const auto& g : gold_selected_unique) {
		BOOST_TEST_CONTEXT("gold value  = " << g) {
			const bool p = contender_selected_unique.find(g) !=
			               contender_selected_unique.end();
			BOOST_TEST(p);
		}
	}
}

BOOST_DATA_TEST_CASE(test_k_selection_int, int_test_datas, td) {
	do_k_selection_test(td);
}

BOOST_DATA_TEST_CASE(test_k_selection_float, float_test_datas, td) {
	do_k_selection_test(td);
}
