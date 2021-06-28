#include <algorithm>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/memory-resource.h>

namespace bdata = boost::unit_test::data;
namespace utf = boost::unit_test;
using namespace thrustshift;

namespace {

template <class MemoryResource>
std::pmr::vector<float> construct(MemoryResource& mres) {
	const int N = 2401;
	std::pmr::vector<float> v0(2 * N, &mres);
	std::pmr::vector<float> v(N, &mres);
	return v;
}

} // namespace

// Closed Issue #2
BOOST_AUTO_TEST_CASE(test_memory_resource_with_pmr_vector_construction_cpu,
                     *boost::unit_test::disabled()) {
	pmr::delayed_pool_type<pmr::host_resource_type> memory_resource;
	{ auto v = construct(memory_resource); }

	for (const auto [k, v] : memory_resource.get_book()) {
		for (const auto& page : v) {
			BOOST_TEST(!page.allocated);
			using T = float;
			const size_t N = k.bytes / sizeof(T);
			std::vector<T> dst(N);
			gsl_lite::span<T> src(reinterpret_cast<T*>(page.ptr), N);
			for (size_t i = 0; i < N; ++i) {
				dst[i] = src[i];
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(test_pool_size_memory_resource) {
	pmr::delayed_pool_type<pmr::host_resource_type> memory_resource;
	[[maybe_unused]] auto p0 = memory_resource.allocate(10, 4);
	[[maybe_unused]] auto p1 = memory_resource.allocate(10, 8);
	[[maybe_unused]] auto p2 = memory_resource.allocate(20, 8);
	[[maybe_unused]] auto p3 = memory_resource.allocate(20, 4);
	BOOST_TEST(memory_resource.get_book().size() == 4);
}

BOOST_AUTO_TEST_CASE(test_fragmenting_pool_size_memory_resource) {
	pmr::delayed_fragmenting_pool_type<pmr::host_resource_type> memory_resource;
	[[maybe_unused]] auto p0 = memory_resource.allocate(10, 4);
	[[maybe_unused]] auto p1 = memory_resource.allocate(10, 8);
	[[maybe_unused]] auto p2 = memory_resource.allocate(20, 8);
	[[maybe_unused]] auto p3 = memory_resource.allocate(20, 4);
	BOOST_TEST(memory_resource.get_book().size() == 4);
	memory_resource.deallocate(p0, 10, 4);
	[[maybe_unused]] auto p4 = memory_resource.allocate(8, 4);
	BOOST_TEST(memory_resource.get_book().size() == 4);
	BOOST_TEST(p4 == p0);
}
