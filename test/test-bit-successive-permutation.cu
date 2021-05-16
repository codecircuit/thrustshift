#include <iostream>
#include <numeric>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/permutation.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

namespace {

struct test_data_t {
	std::vector<bool> swap_flags;
};

inline std::ostream& operator<<(std::ostream& os, const test_data_t& td) {

	os << "swap_flags = ";
	for (auto ds : td.swap_flags) {
		os << ds << ", ";
	}
	return os;
}

std::vector<test_data_t> test_data = {
    // clang-format off
	{{false, false, false, false}},
	{{false, false, true, false}},
	{{false, true, true, false}},
	{{true, false, false, false}},
	{{false, true, false, false, false}},
	{{false, false, true, false, false, false}},
	{{false, false, true, true, true, false}},
	{{false, false, true, true, true, true}},
	{{true, true, true, true}},
	{{true, true, true, false}},
	{{false, true, true, true}},
	{{false}},
	{{true}},
    // clang-format on
};

inline auto calc_gold_permutation(const std::vector<bool>& swap_flags) {
	const int N = swap_flags.size() + 1;
	std::vector<int> permutation(N);
	std::iota(permutation.begin(), permutation.end(), 0);
	for (int j = 0, e = swap_flags.size(); j < e; ++j) {
		if (swap_flags[j]) {
			std::swap(permutation[j + 1], permutation[j]);
		}
	}
	return permutation;
}

} // namespace

BOOST_DATA_TEST_CASE(test_bit_successive_permutation, test_data, td) {
	using BitPatternT = long long unsigned;
	const int N = td.swap_flags.size() + 1;
	permutation::bit_successive_permutation_t<BitPatternT> bsp(N);
	for (int j = 0, e = td.swap_flags.size(); j < e; ++j) {
		bsp.set(j, td.swap_flags[j]);
	}
	std::vector<int> permutation(N);
	for (int i = N - 1; i >= 0; --i) {
		permutation[i] = bsp[i];
	}
	auto gold_permutation = calc_gold_permutation(td.swap_flags);
	for (int i = 0; i < N; ++i) {
		BOOST_TEST_CONTEXT("i = "
		                   << i << ", gold_perm[i] = " << gold_permutation[i]
		                   << ", contender_perm[i] = " << permutation[i]) {
			BOOST_TEST(gold_permutation[i] == permutation[i]);
		}
	}
}
