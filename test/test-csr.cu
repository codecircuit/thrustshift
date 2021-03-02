#include <algorithm>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <thrustshift/CSR.h>
#include <thrustshift/eigen3>

namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_CASE(test_csr_compilation) {
	thrustshift::CSR<float, int> csr(std::vector<float>{1, 2, 3},
	                                 std::vector<int>{4, 5, 6},
	                                 std::vector<int>{0, 3},
	                                 7);
	thrustshift::CSR_view<float, int> view(csr);
	[[maybe_unused]] gsl_lite::span<float> s0 = view.values();
	thrustshift::CSR_view<const float, int> view2(csr);
	[[maybe_unused]] gsl_lite::span<const float> s1 = view2.values();
}

BOOST_AUTO_TEST_CASE(test_csr_conversion) {
	using EMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

	{
		thrustshift::CSR<float, int> csr(std::vector<float>{1, 2, 3},
		                                 std::vector<int>{4, 5, 6},
		                                 std::vector<int>{0, 1, 3},
		                                 7);
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		BOOST_TEST(eig_mtx.coeff(0, 4) == 1);
		BOOST_TEST(eig_mtx.coeff(1, 5) == 2);
		BOOST_TEST(eig_mtx.coeff(1, 6) == 3);
	}
	{
		thrustshift::CSR<float, int> csr(std::vector<float>{0, 0, 0},
		                                 std::vector<int>{4, 5, 6},
		                                 std::vector<int>{0, 1, 3},
		                                 7);
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		BOOST_TEST(eig_mtx.coeff(0, 4) == 0);
		BOOST_TEST(eig_mtx.coeff(1, 5) == 0);
		BOOST_TEST(eig_mtx.coeff(1, 6) == 0);
	}
	{
		thrustshift::CSR<float, int> csr(
		    std::vector<float>{}, std::vector<int>{}, std::vector<int>{0}, 7);
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
	}
}

BOOST_AUTO_TEST_CASE(test_csr_extend_rows) {
	{
		thrustshift::CSR<float, int> csr(std::vector<float>{1, 2, 3},
		                                 std::vector<int>{4, 5, 6},
		                                 std::vector<int>{0, 1, 3},
		                                 7);
		using EMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		csr.extend_rows(1, 0);

		BOOST_TEST(csr.values()[0] == 0);
		BOOST_TEST(csr.values()[1] == 1);
		BOOST_TEST(csr.values()[2] == 0);
		BOOST_TEST(csr.values()[3] == 2);
		BOOST_TEST(csr.values()[4] == 3);
		BOOST_TEST(csr.col_indices()[0] == 0);
		BOOST_TEST(csr.col_indices()[1] == 4);
		BOOST_TEST(csr.col_indices()[2] == 0);
		BOOST_TEST(csr.col_indices()[3] == 5);
		BOOST_TEST(csr.col_indices()[4] == 6);
		BOOST_TEST(csr.row_ptrs()[0] == 0);
		BOOST_TEST(csr.row_ptrs()[1] == 2);
		BOOST_TEST(csr.row_ptrs()[2] == 5);
		BOOST_TEST(csr.values().size() == size_t(5));
		BOOST_TEST(csr.col_indices().size() == size_t(5));
		BOOST_TEST(csr.row_ptrs().size() == size_t(3));
		EMatrix eig_mtx_ =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		const bool b = eig_mtx == eig_mtx_;
		BOOST_TEST(b);
	}
	{
		thrustshift::CSR<float, int> csr(std::vector<float>{},
		                                 std::vector<int>{},
		                                 std::vector<int>{0, 0, 0, 0, 0},
		                                 4);
		using EMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		csr.extend_rows(20, 0);
		BOOST_TEST(csr.values().size() == size_t(16));
		BOOST_TEST(csr.col_indices()[0] == 0);
		BOOST_TEST(csr.col_indices()[1] == 1);
		BOOST_TEST(csr.col_indices()[2] == 2);
		BOOST_TEST(csr.col_indices()[3] == 3);
		BOOST_TEST(csr.col_indices()[4] == 0);
		for (const auto e : csr.values()) {
			BOOST_TEST(e == 0);
		}

		EMatrix eig_mtx_ =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		const bool b = eig_mtx == eig_mtx_;
		BOOST_TEST(b);
	}
	{
		thrustshift::CSR<float, int> csr(std::vector<float>{},
		                                 std::vector<int>{},
		                                 std::vector<int>{0, 0, 0, 0, 0},
		                                 4);
		using EMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		csr.extend_rows(2, 0);
		BOOST_TEST(csr.values().size() == size_t(8));
		BOOST_TEST(csr.col_indices()[0] == 0);
		BOOST_TEST(csr.col_indices()[1] == 1);
		BOOST_TEST(csr.col_indices()[2] == 0);
		BOOST_TEST(csr.col_indices()[3] == 1);
		BOOST_TEST(csr.col_indices()[4] == 0);
		BOOST_TEST(csr.col_indices()[5] == 1);
		BOOST_TEST(csr.col_indices()[6] == 0);
		BOOST_TEST(csr.col_indices()[7] == 1);
		BOOST_TEST(csr.row_ptrs()[0] == 0);
		BOOST_TEST(csr.row_ptrs()[1] == 2);
		BOOST_TEST(csr.row_ptrs()[2] == 4);
		BOOST_TEST(csr.row_ptrs()[3] == 6);
		BOOST_TEST(csr.row_ptrs()[4] == 8);

		for (const auto e : csr.values()) {
			BOOST_TEST(e == 0);
		}

		EMatrix eig_mtx_ =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		const bool b = eig_mtx == eig_mtx_;
		BOOST_TEST(b);
	}
	{
		thrustshift::CSR<float, int> csr(std::vector<float>{7, 8, 9},
		                                 std::vector<int>{1, 2, 3},
		                                 std::vector<int>{0, 0, 3, 3, 3},
		                                 4);
		using EMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
		EMatrix eig_mtx =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		csr.extend_rows(2, 0);
		BOOST_TEST(csr.values().size() == size_t(10));
		BOOST_TEST(csr.col_indices()[0] == 0);
		BOOST_TEST(csr.col_indices()[1] == 1);
		BOOST_TEST(csr.col_indices()[2] == 0);
		BOOST_TEST(csr.col_indices()[3] == 1);
		BOOST_TEST(csr.col_indices()[4] == 2);
		BOOST_TEST(csr.col_indices()[5] == 3);
		BOOST_TEST(csr.col_indices()[6] == 0);
		BOOST_TEST(csr.col_indices()[7] == 1);
		BOOST_TEST(csr.col_indices()[8] == 0);
		BOOST_TEST(csr.col_indices()[9] == 1);

		BOOST_TEST(csr.row_ptrs()[0] == 0);
		BOOST_TEST(csr.row_ptrs()[1] == 2);
		BOOST_TEST(csr.row_ptrs()[2] == 6);
		BOOST_TEST(csr.row_ptrs()[3] == 8);
		BOOST_TEST(csr.row_ptrs()[4] == 10);

		EMatrix eig_mtx_ =
		    thrustshift::eigen::csr2sparse_mtx<Eigen::SparseMatrix<float>>(csr);
		const bool b = eig_mtx == eig_mtx_;
		BOOST_TEST(b);
	}

}
