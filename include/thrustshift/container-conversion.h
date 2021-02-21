#pragma once

#include <thrustshift/COO.h>
#include <thrustshift/CSR.h>

namespace thrustshift {

template <class COO_C, class CSR_C, class MemoryResource>
COO_C csr2coo(CSR_C&& csr, MemoryResource& memory_resource) {

	const size_t nnz = csr.values().size();
	COO_C coo(nnz, csr.num_rows(), csr.num_cols(), memory_resource);
	auto device = cuda::device::current::get();
	auto stream = device.default_stream();
	async::copy(stream, csr.values(), coo.values());
	async::copy(stream, csr.col_indices(), coo.col_indices());

	for (size_t row_id = 0; row_id < csr.row_ptrs().size() - 1; ++row_id) {
		// nns = not null space
		for (size_t nns_id = csr.row_ptrs()[row_id];
		     nns_id < csr.row_ptrs()[row_id + 1];
		     ++nns_id) {
			coo.row_indices()[nns_id] = row_id;
		}
	}

	coo.change_storage_order(storage_order_t::row_major);
	stream.synchronize();
	return coo;
}

template <class CSR_C, class COO_C, class MemoryResource>
CSR_C coo2csr(COO_C&& coo_, MemoryResource& memory_resource) {

	if (coo_.get_storage_order() != storage_order_t::row_major) {
		// Here we copy the matrix because we change the storage type
		typename std::remove_cv<std::remove_reference<COO_C>::type>::type coo(
		    std::forward<COO_C>(coo_));
		coo.change_storage_order(storage_order_t::row_major);
		auto row_ptrs = coo.get_ptrs();
		return CSR_C(
		    coo.values(), coo.col_indices(), row_ptrs, memory_resource);
	}
	auto row_ptrs = coo_.get_ptrs();
	return CSR_C(coo_.values(), coo_.col_indices(), row_ptrs, memory_resource);
}

} // namespace thrustshift
