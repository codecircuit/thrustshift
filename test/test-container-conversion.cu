#include <algorithm>
#include <type_traits>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <Eigen/Sparse>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <matrixmarket/eigen.hpp>

#include <thrustshift/COO.h>
#include <thrustshift/CSR.h>
#include <thrustshift/container-conversion.h>
#include <thrustshift/equal.h>
#include <thrustshift/eigen3>

namespace bdata = boost::unit_test::data;

using namespace thrustshift;

namespace {

struct test_data_t {
	std::vector<float> values;
	std::vector<int> row_indices;
	std::vector<int> col_indices;
	size_t num_rows;
	size_t num_cols;
};

inline std::ostream& operator<<(std::ostream& os,
                                [[maybe_unused]] const test_data_t& td) {
	return os;
}

const std::vector<test_data_t> test_data = {
    // clang-format off
	{
		{1, 2, 3, 4},
		{0, 1, 2, 3},
		{1, 2, 3, 4},
		7,
		7
	},
	{
		{1},
		{3},
		{3},
		5,
		6
	},
	{
		{1, 1, 1},
		{3, 3, 3},
		{3, 3, 3},
		5,
		6
	},
	{
		{},
		{},
		{},
		0,
		0
	},
	{
		{0},
		{0},
		{0},
		1,
		1
	},
	{
		{1, 2, 3, 4},
		{0, 0, 0, 3},
		{1, 2, 3, 3},
		4,
		4
	},
	{
		{1, 2, 3, 4},
		{0, 0, 0, 3},
		{3, 2, 1, 3},
		4,
		4
	},

    // clang-format on
};

} // namespace

BOOST_DATA_TEST_CASE(test_container_conversion, test_data, td) {
	thrustshift::COO<float, int> coo(
	    td.values, td.row_indices, td.col_indices, td.num_rows, td.num_cols);
	coo.change_storage_order(thrustshift::storage_order_t::row_major);
	pmr::delayed_pool_type<pmr::managed_resource_type> delayed_memory_resource;
	auto csr = coo2csr<CSR<float, int>>(coo, delayed_memory_resource);
	auto coo2 = csr2coo<COO<float, int>>(csr, delayed_memory_resource);

	const bool b = coo2 == coo;
	BOOST_TEST(b);
}

BOOST_AUTO_TEST_CASE(test_container_conversion_matrix_store) {
	const std::string pth =
	    "/mnt/matrix_store/ModelPDEs/A_problem4n256square/"
	    "A_problem4n256square.mtx";
	Eigen::SparseMatrix<float> eigen_mtx =
	    matrixmarket::readEigenSparseFromFile<float>(pth);
	pmr::delayed_pool_type<pmr::managed_resource_type> delayed_memory_resource;
	auto csr = eigen::sparse_mtx2csr(eigen_mtx);
	auto coo = csr2coo<COO<float, int>>(csr, delayed_memory_resource);
	auto csr2 = coo2csr<CSR<float, int>>(coo, delayed_memory_resource);
	const bool b = csr == csr2;
	BOOST_TEST(b);
}

BOOST_AUTO_TEST_CASE(test_tuple_to_array_conversion) {

	std::tuple<double, double, double> tup(1, 2, 3);

	auto arr = tuple2kat_array(tup);
	BOOST_TEST(arr[0] == 1.0);
	BOOST_TEST(arr[1] == 2.0);
	BOOST_TEST(arr[2] == 3.0);
	BOOST_TEST(arr.size() == 3);
	using ArrayT = decltype(arr);
	constexpr bool b = std::is_same<double, ArrayT::value_type>::value;
	BOOST_TEST(b);
}
