#include <set>
#include <vector>
#include <algorithm>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>


#include <thrustshift/COO.h>

namespace {

namespace bdata = boost::unit_test::data;

struct test_data_t {
	std::vector<float> values;
	std::vector<int> row_indices;
	std::vector<int> col_indices;
	std::vector<int> gold_row_ptrs;
	std::vector<int> gold_col_ptrs;
	size_t num_rows;
	size_t num_cols;
};

inline std::ostream& operator<<(std::ostream& os, const test_data_t& td) {
	return os;
}

const std::vector<test_data_t> test_data = {
    // clang-format off
	{
		{1, 2, 3, 4},
		{0, 1, 2, 3},
		{1, 2, 3, 4},
		{0, 1, 2, 3, 4, 4, 4, 4},
		{0, 0, 1, 2, 3, 4, 4, 4},
		7,
		7
	},
	{
		{1},
		{3},
		{3},
		{0, 0, 0, 0, 1, 1},
		{0, 0, 0, 0, 1, 1, 1},
		5,
		6
	},
	{
		{1, 1, 1},
		{3, 3, 3},
		{3, 3, 3},
		{0, 0, 0, 0, 3, 3},
		{0, 0, 0, 0, 3, 3, 3},
		5,
		6
	},
	{
		{},
		{},
		{},
		{0},
		{0},
		0,
		0
	},
	{
		{0},
		{0},
		{0},
		{0, 1},
		{0, 1},
		1,
		1
	},
	{
		{1, 2, 3, 4},
		{0, 0, 0, 3},
		{1, 2, 3, 3},
		{0, 3, 3, 3, 4},
		{0, 0, 1, 2, 4},
		4,
		4
	},

    // clang-format on
};

} // namespace


BOOST_DATA_TEST_CASE(test_coo, test_data, td) {
	thrustshift::COO<float, int> coo(td.values, td.row_indices, td.col_indices, td.num_rows, td.num_cols);
	coo.change_storage_order(thrustshift::storage_order_t::row_major);
	auto row_ptrs = coo.get_ptrs();
	BOOST_TEST(row_ptrs.size() == td.gold_row_ptrs.size());
	BOOST_TEST(std::equal(row_ptrs.begin(), row_ptrs.end(), td.gold_row_ptrs.begin()));

	coo.change_storage_order(thrustshift::storage_order_t::col_major);
	auto col_ptrs = coo.get_ptrs();
	BOOST_TEST(col_ptrs.size() == td.gold_col_ptrs.size());
	BOOST_TEST(std::equal(col_ptrs.begin(), col_ptrs.end(), td.gold_col_ptrs.begin()));
}
