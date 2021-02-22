#pragma once

#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include <Eigen/Sparse>

#include <thrustshift/CSR.h>
#include <thrustshift/managed-vector.h>

namespace thrustshift {

namespace eigen {

//! Return a CSR matrix with the default memory resource of the CSR class
template <class EigenSparseMatrix>
auto sparse_mtx2csr(EigenSparseMatrix&& m_) {

	using DataType =
	    typename std::remove_reference<EigenSparseMatrix>::type::value_type;
	using StorageIndex =
	    typename std::remove_reference<EigenSparseMatrix>::type::StorageIndex;
	// Create copy because we might modify the container with `makeCompressed`
	Eigen::SparseMatrix<DataType, Eigen::RowMajor, StorageIndex> m = m_;
	m.makeCompressed();
	auto m_data = m.data();
	auto nnz = m.nonZeros();
	auto rows = m.rows();

	const DataType* A = &m_data.value(0);
	const StorageIndex* IA = m.outerIndexPtr();
	const StorageIndex* JA = &m_data.index(0);

	managed_vector<DataType> seq_A(A, A + nnz);
	managed_vector<StorageIndex> seq_JA(JA, JA + nnz);
	managed_vector<StorageIndex> seq_IA(IA, IA + rows + 1);

	return CSR<DataType,StorageIndex>(seq_A, seq_JA, seq_IA, gsl_lite::narrow<size_t>(m.cols()));
}

} // namespace eigen

} // namespace thrustshift
