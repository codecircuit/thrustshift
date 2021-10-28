#pragma once

#include <cuda/runtime_api.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cub/cub.cuh>

#include <thrustshift/math.h>
#include <thrustshift/not-a-vector.h>

namespace thrustshift {

namespace async {

template <class ValuesRange,
          class SelectedRange,
          class NumSelectedPtr,
          class SelectOp,
          class MemoryResource>
void select_if(cuda::stream_t& stream,
               ValuesRange&& values,
               SelectedRange&& selected,
               NumSelectedPtr num_selected_ptr,
               SelectOp select_op,
               MemoryResource& delayed_memory_resource) {

	const std::size_t N = values.size();

	gsl_Expects(selected.size() == N);

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;

	using T = typename std::remove_reference<ValuesRange>::type::value_type;

	auto exec = [&] {
		cuda::throw_if_error(cub::DeviceSelect::If(tmp_ptr,
		                                           tmp_bytes_size,
		                                           values.data(),
		                                           selected.data(),
		                                           num_selected_ptr,
		                                           N,
		                                           select_op,
		                                           stream.id()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

/*! \brief Select values based on predicate with their index.
 *
 *  \param values Range of length N with value_type `T`
 *  \param selected Range of length N with a value type which is assignable
 *      by a thrust::tuple<T, int>.
 *  \param select_op select predicate with signature
 *      ```
 *      select_op = [] __device__ (const thrust::tuple<T, int>& tup) { ... };
 *      ```
 */
template <class ValuesRange,
          class SelectedRange,
          class NumSelectedPtr,
          class SelectOp,
          class MemoryResource>
void select_if_with_index(cuda::stream_t& stream,
                          ValuesRange&& values,
                          SelectedRange&& selected,
                          NumSelectedPtr num_selected_ptr,
                          SelectOp select_op,
                          MemoryResource& delayed_memory_resource) {

	const std::size_t N = values.size();

	gsl_Expects(selected.size() == N);

	size_t tmp_bytes_size = 0;
	void* tmp_ptr = nullptr;

	using T = typename std::remove_reference<ValuesRange>::type::value_type;

	auto cit = thrust::make_counting_iterator(0);
	auto it = thrust::make_zip_iterator(thrust::make_tuple(values.data(), cit));

	auto exec = [&] {
		cuda::throw_if_error(cub::DeviceSelect::If(tmp_ptr,
		                                           tmp_bytes_size,
		                                           it,
		                                           selected.data(),
		                                           num_selected_ptr,
		                                           N,
		                                           select_op,
		                                           stream.id()));
	};
	exec();
	auto tmp =
	    make_not_a_vector<uint8_t>(tmp_bytes_size, delayed_memory_resource);
	tmp_ptr = tmp.to_span().data();
	exec();
}

} // namespace async

} // namespace thrustshift
