#include <algorithm>
#include <iostream>
#include <variant>
#include <vector>

#include <Eigen/Sparse>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/memory-resource.h>
#include <thrustshift/wrap-subgroups.h>
#include <thrustshift/managed-vector.h>

namespace bdata = boost::unit_test::data;
using namespace thrustshift;

namespace {

struct test_data_t {
	std::vector<int> subgroup_ptrs;
	int mean_group_size;
	std::vector<int> gold_group_ptrs;
};

std::ostream& operator<<(std::ostream& os, const test_data_t& td) {

	os << "subgroup_ptrs = ";
	for (auto i : td.subgroup_ptrs) {
		os << i << ", ";
	}
	os << "\ngold_group_ptrs = ";
	for (auto i : td.gold_group_ptrs) {
		os << i << ", ";
	}
	os << "\nmean_group_size = " << td.mean_group_size;

	return os;
}

std::vector<test_data_t> test_data = {
	// clang-format off
	{
		{0, 5, 10, 15, 20, 25, 30, 35, 40, 45},
		10,
		{0, 2, 4, 6, 9}
	},
	{
		{0, 45},
		10,
		{0, 1}
	},
	{
		{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 45, 45, 45},
		10,
		{0, 2, 4, 6, 12}
	},
	{
		{0, 45, 45, 45, 45},
		10,
		{0, 4}
	},
	// clang-format on
};

}

BOOST_DATA_TEST_CASE(test_wrap_subgroups, test_data, td) {

	const int mean_group_size = td.mean_group_size;
	const int num_elements = td.subgroup_ptrs.back();

	managed_vector<int> subgroup_ptrs(td.subgroup_ptrs.begin(), td.subgroup_ptrs.end());
	managed_vector<int> group_ptrs(std::max(num_elements / mean_group_size + 1, 2));

	auto group_ptrs_size_ = cuda::memory::managed::make_unique<int>();
	auto group_ptrs_size = gsl_lite::make_not_null<int*>(group_ptrs_size_.get());

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();

	auto memory_resource = pmr::delayed_pool_type<pmr::managed_resource_type>();

	async::wrap_subgroups<int>(stream, subgroup_ptrs, num_elements, group_ptrs,mean_group_size, group_ptrs_size, memory_resource );
	device.synchronize();

	BOOST_TEST(*group_ptrs_size == td.gold_group_ptrs.size());
	for (int i = 0; i < *group_ptrs_size; ++i) {
		BOOST_TEST(group_ptrs[i] == td.gold_group_ptrs[i]);
	}
}
