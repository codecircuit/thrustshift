#pragma once

#include <utility>

#include <thrust/tuple.h>

#include <cuda/std/array>

#include <thrustshift/COO.h>
#include <thrustshift/CSR.h>
#include <thrustshift/copy.h>
#include <thrustshift/defines.h>
#include <thrustshift/type-traits.h>

namespace thrustshift {

template <class RowPtrsRange, class RowIndicesRange>
void row_ptrs2row_indices(RowPtrsRange&& row_ptrs,
                          RowIndicesRange&& row_indices) {
	gsl_Expects(!row_ptrs.empty());
	for (size_t row_id = 0; row_id < row_ptrs.size() - 1; ++row_id) {
		// nns = not null space
		for (auto nns_id = row_ptrs[row_id]; nns_id < row_ptrs[row_id + 1];
		     ++nns_id) {
			row_indices[nns_id] = row_id;
		}
	}
}

/*! \brief Convert a CSR container to a COO container
 *
 *  Usage example:
 *
 *  ```
 *  auto coo = csr2coo<thrustshift::COO<double, float>>(csr_matrix, memory_resource);
 *  ```
 *
 *  The template parameter is not deduced implicitly because type conversions are
 *  allowed. E.g., you can create a double precision COO matrix from a single
 *  precision CSR matrix.
 */
template <class COO_C, class CSR_C, class MemoryResource>
COO_C csr2coo(CSR_C&& csr, MemoryResource& memory_resource) {

	const size_t nnz = csr.values().size();
	COO_C coo(nnz, csr.num_rows(), csr.num_cols(), memory_resource);
	cudaStream_t stream = 0;
	async::copy(stream, csr.values(), coo.values());
	async::copy(stream, csr.col_indices(), coo.col_indices());

	row_ptrs2row_indices(csr.row_ptrs(), coo.row_indices());

	coo.change_storage_order(storage_order_t::row_major);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	return coo;
}

/*! \brief Convert a COO container to a CSR container
 *
 *  Usage example:
 *
 *  ```
 *  auto csr = coo2csr<thrustshift::CSR<double, float>>(coo_matrix, memory_resource);
 *  ```
 *
 *  The template parameter is not deduced implicitly because type conversions are
 *  allowed. E.g., you can create a double precision CSR matrix from a single
 *  precision COO matrix.
 */
template <class CSR_C, class COO_C, class MemoryResource>
CSR_C coo2csr(COO_C&& coo_, MemoryResource& memory_resource) {

	if (coo_.get_storage_order() != storage_order_t::row_major) {
		// Here we copy the matrix because we change the storage type
		typename std::remove_cv<typename std::remove_reference<COO_C>::type>::
		    type coo(std::forward<COO_C>(coo_));
		coo.change_storage_order(storage_order_t::row_major);
		auto row_ptrs = coo.get_ptrs();
		return CSR_C(coo.values(),
		             coo.col_indices(),
		             row_ptrs,
		             coo_.num_cols(),
		             memory_resource);
	}
	auto row_ptrs = coo_.get_ptrs();
	return CSR_C(coo_.values(),
	             coo_.col_indices(),
	             row_ptrs,
	             coo_.num_cols(),
	             memory_resource);
}

namespace detail {

template <class ArrayT, std::size_t... I>
THRUSTSHIFT_FHD auto array2thrust_tuple_impl(const ArrayT& arr,
                                             std::index_sequence<I...>) {
	return thrust::make_tuple(arr[I]...);
}

} // namespace detail

template <typename T,
          std::size_t N,
          template <typename, std::size_t>
          class Arr,
          class I = std::make_index_sequence<N>>
THRUSTSHIFT_FHD auto array2thrust_tuple(const Arr<T, N>& arr) {
	return detail::array2thrust_tuple_impl(arr, I{});
}

namespace detail {

template <class Tuple, std::size_t... I>
THRUSTSHIFT_FHD auto tuple2cuda_array_impl(const Tuple& tup,
                                           std::index_sequence<I...>) {
	using std::get;
	// https://stackoverflow.com/questions/25732386/what-is-stddecay-and-when-it-should-be-used#:~:text=It%20is%20used%20in%20the,is%20stored%20and%20so%20on.
	// `std::decay(decltype(get<0>(tup))::type` can be used to deduce the type of
	// `auto value = std::get<0>(tup)`
	using T = typename std::decay<decltype(get<0>(tup))>::type;
	constexpr std::size_t N = sizeof(Tuple) / sizeof(T);
	return cuda::std::array<T, N>({get<I>(tup)...});
}

} // namespace detail

//! Convert a tuple of equal types `T` into a cuda array.
//! If the tuple has different types, the behaviour is undefined.
template <typename T, typename... Rest, template <typename...> class Tuple>
THRUSTSHIFT_FHD auto tuple2cuda_array(const Tuple<T, Rest...>& tup) {
	constexpr std::size_t len = sizeof(Tuple<T, Rest...>) / sizeof(T);
	using I = std::make_index_sequence<len>;
	return detail::tuple2cuda_array_impl(tup, I{});
}

} // namespace thrustshift
