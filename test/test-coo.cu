#include <algorithm>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <Eigen/Sparse>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <matrixmarket/eigen.hpp>

#include <thrustshift/COO.h>
#include <thrustshift/eigen3>

#include "./memory-resource-check.h"

namespace bdata = boost::unit_test::data;

using namespace thrustshift;

namespace {

struct test_data_t {
	std::vector<float> values;
	std::vector<int> row_indices;
	std::vector<int> col_indices;
	std::vector<int> gold_row_ptrs;
	std::vector<int> gold_col_ptrs;
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
		{2, 3, 4},
		{2, 3, 4},
		{0, 0, 0, 1, 2, 3},
		{0, 0, 0, 1, 2, 3, 3},
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
	{
		{1, 2, 3, 4},
		{0, 0, 0, 3},
		{3, 2, 1, 3},
		{0, 3, 3, 3, 4},
		{0, 0, 1, 2, 4},
		4,
		4
	},

    // clang-format on
};

} // namespace

BOOST_DATA_TEST_CASE(test_coo, test_data, td) {
	thrustshift::COO<float, int> coo(
	    td.values, td.row_indices, td.col_indices, td.num_rows, td.num_cols);
	coo.change_storage_order(thrustshift::storage_order_t::row_major);
	auto row_ptrs = coo.get_ptrs();
	BOOST_TEST(row_ptrs.size() == td.gold_row_ptrs.size());
	BOOST_TEST(
	    std::equal(row_ptrs.begin(), row_ptrs.end(), td.gold_row_ptrs.begin()));

	coo.change_storage_order(thrustshift::storage_order_t::col_major);
	auto col_ptrs = coo.get_ptrs();
	BOOST_TEST(col_ptrs.size() == td.gold_col_ptrs.size());
	BOOST_TEST(
	    std::equal(col_ptrs.begin(), col_ptrs.end(), td.gold_col_ptrs.begin()));
}

BOOST_AUTO_TEST_CASE(test_coo_ctors) {

	thrustshift::COO<float, int> coo(10, 10, 10);
	thrustshift::COO<float, int> coo2(coo);
	thrustshift::COO<float, int> coo3 = coo;
	auto coo4 = coo;
	thrustshift::COO_view<float, int> view0(coo);
	[[maybe_unused]] gsl_lite::span<float> s0 = view0.values();
	thrustshift::COO_view<const float, int> view1(coo);
	[[maybe_unused]] gsl_lite::span<const float> s1 = view1.values();
	[[maybe_unused]] gsl_lite::span<int> s2 = view1.col_indices();
	thrustshift::COO_view<const float, const int> view2(coo);
}

BOOST_DATA_TEST_CASE(test_symmetrize_abs_coo, test_data, td) {

	// Symmetrize test are undefined on non-square matrices
	if (td.num_rows == td.num_cols) {
		thrustshift::COO<float, int> coo(td.values,
		                                 td.row_indices,
		                                 td.col_indices,
		                                 td.num_rows,
		                                 td.num_cols);

		auto eigen_mtx = eigen::coo2sparse_mtx<Eigen::SparseMatrix<float>>(coo);
		eigen_mtx =
		    eigen_mtx.cwiseAbs() +
		    Eigen::SparseMatrix<float>(eigen_mtx.transpose()).cwiseAbs();
		auto gold_coo_sym = eigen::sparse_mtx2coo(eigen_mtx);
		pmr::delayed_pool_type<pmr::managed_resource_type> memory_resource;
		auto coo_sym = symmetrize_abs<float, int>(coo, memory_resource);
		const bool b = coo_sym == gold_coo_sym;
		BOOST_TEST(b);
	}
}

// Disabled by default is this requires a matrix market file
BOOST_AUTO_TEST_CASE(test_symmetrize_abs_coo_matrix_store,
                     *boost::unit_test::disabled()) {

	const std::string pth =
	    "/mnt/matrix_store/ModelPDEs/A_problem4n256square/"
	    "A_problem4n256square.mtx";
	Eigen::SparseMatrix<float> eigen_mtx =
	    matrixmarket::readEigenSparseFromFile<float>(pth);

	// Symmetrize test are undefined on non-square matrices
	if (eigen_mtx.rows() == eigen_mtx.cols()) {

		pmr::delayed_pool_type<pmr::managed_resource_type> memory_resource;
		{
			auto coo = eigen::sparse_mtx2coo(eigen_mtx);

			eigen_mtx =
			    eigen_mtx.cwiseAbs() +
			    Eigen::SparseMatrix<float>(eigen_mtx.transpose()).cwiseAbs();
			auto gold_coo_sym = eigen::sparse_mtx2coo(eigen_mtx);
			auto coo_sym = symmetrize_abs<float, int>(coo, memory_resource);
			const bool b = coo_sym == gold_coo_sym;
			BOOST_TEST(b);
		}

		touch_all_memory_resource_pages(memory_resource);
	}
}

BOOST_DATA_TEST_CASE(test_coo_get_diagonal, test_data, td) {

	cudaStream_t stream = 0;

	// Symmetrize test are undefined on non-square matrices
	thrustshift::COO<float, int> coo(
	    td.values, td.row_indices, td.col_indices, td.num_rows, td.num_cols);

	const size_t N = std::min(coo.num_rows(), coo.num_cols());

	thrustshift::managed_vector<float> diagonal(N);

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_mtx =
	    eigen::coo2sparse_mtx<Eigen::SparseMatrix<float>>(coo);
	async::get_diagonal<float>(stream, coo, diagonal);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	auto eig_diag = eigen_mtx.diagonal();
	for (size_t i = 0; i < N; ++i) {
		BOOST_TEST_CONTEXT("row_id = " << i << ", eigen_diag = " << eig_diag[i]
		                               << ", thrustshift_diag = "
		                               << diagonal[i]) {
			BOOST_TEST(eig_diag[i] == diagonal[i]);
		}
	}
}
