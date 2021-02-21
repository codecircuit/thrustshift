#include <vector>
#include <algorithm>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/CSR.h>

namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_CASE(test_csr_compilation) {
	thrustshift::CSR<float, int> csr(std::vector<float>{1,2,3}, std::vector<int>{4,5,6}, std::vector<int>{0, 3}, 7);
	thrustshift::CSR_view<float, int> view(csr);
	[[maybe_unused]] gsl_lite::span<float> s0 = view.values();
	thrustshift::CSR_view<const float, int> view2(csr);
	[[maybe_unused]] gsl_lite::span<const float> s1 = view2.values();
}
