#pragma once

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

	thrustshift::pmr::managed_resource_type memory_resource;
	return csr2coo<thrustshift::COO<DataType, StorageIndex>>(
	    sparse_mtx2csr(std::forward<EigenSparseMatrix>(m_)), memory_resource);
}

} // namespace eigen

} // namespace thrustshift
