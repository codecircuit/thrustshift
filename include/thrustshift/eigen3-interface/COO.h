#pragma once

#include <vector>

#include <Eigen/Sparse>

#include <thrustshift/container-conversion.h>
#include <thrustshift/eigen3-interface/CSR.h>
#include <thrustshift/memory-resource.h>

namespace thrustshift {

namespace eigen {

//! Return a COO matrix with managed memory
template <class EigenSparseMatrix>
auto sparse_mtx2coo(EigenSparseMatrix&& m_) {

	using DataType =
	    typename std::remove_reference<EigenSparseMatrix>::type::value_type;
	using StorageIndex =
	    typename std::remove_reference<EigenSparseMatrix>::type::StorageIndex;

	return csr2coo<thrustshift::COO<DataType, StorageIndex>>(
	    sparse_mtx2csr(std::forward<EigenSparseMatrix>(m_)), pmr::default_resource);
}

template <class EigenSparseMatrix, class COO_C>
EigenSparseMatrix coo2sparse_mtx(COO_C&& coo) {

	EigenSparseMatrix mtx(coo.num_rows(), coo.num_cols());
	using DataType = typename EigenSparseMatrix::value_type;
	std::vector<Eigen::Triplet<DataType>> triplets;
	auto values = coo.values();
	auto row_indices = coo.row_indices();
	auto col_indices = coo.col_indices();
	for (size_t nns_id = 0; nns_id < coo.values().size(); ++nns_id) {
		triplets.push_back(
		    {row_indices[nns_id], col_indices[nns_id], values[nns_id]});
	}
	mtx.setFromTriplets(triplets.begin(), triplets.end());
	return mtx;
}

} // namespace eigen

} // namespace thrustshift
