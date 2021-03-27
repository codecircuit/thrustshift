#pragma once

#include <vector>
#include <set>

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

template <class TripletRange>
bool triplets_are_unique(TripletRange&& triplets) {

	std::set<std::array<int, 2>> points;
	for (auto t : triplets) {
		points.insert({t.row(), t.col()});
	}
	return triplets.size() == points.size();
}

template <class EigenSparseMatrix, class COO_C>
EigenSparseMatrix coo2sparse_mtx(COO_C&& coo) {

	EigenSparseMatrix mtx(coo.num_rows(), coo.num_cols());
	using DataType = typename EigenSparseMatrix::value_type;
	std::vector<Eigen::Triplet<DataType>> triplets(coo.values().size());
	auto values = coo.values();
	auto row_indices = coo.row_indices();
	auto col_indices = coo.col_indices();
	for (size_t nns_id = 0; nns_id < coo.values().size(); ++nns_id) {
		const auto row_index = row_indices[nns_id];
		const auto col_index = col_indices[nns_id];
		const auto value = values[nns_id];
		gsl_Expects(row_index >= 0 && row_index < coo.num_rows());
		gsl_Expects(col_index >= 0 && col_index < coo.num_cols());
		triplets[nns_id] = Eigen::Triplet<DataType>(
		    {row_indices[nns_id], col_indices[nns_id], values[nns_id]});
	}
	// Because `setFromTriplets` obeys UB if points occur more than once.
	// That is hard to Debug. Thus we do a check here.
	gsl_Expects(triplets_are_unique(triplets));
	mtx.setFromTriplets(triplets.begin(), triplets.end());
	return mtx;
}

} // namespace eigen

} // namespace thrustshift
