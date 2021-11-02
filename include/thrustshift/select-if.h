#pragma once

#include <type_traits>

#include <cuda/define_specifiers.hpp>
#include <cuda/runtime_api.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <gsl-lite/gsl-lite.hpp>

#include <cub/cub.cuh>

#include <thrustshift/constant.h>
#include <thrustshift/math.h>
#include <thrustshift/not-a-vector.h>

namespace thrustshift {

namespace device_function {

namespace implicit_unroll {

/*! \brief Write the selected values to the result if select_op is true.
 *
 *  All threads check if their value should be selected. Afterwards, it is
 *  determined within the warp how many values were selected. Then, the first
 *  in the warp determines the range in the result, which is reserved for the
 *  warp via an atomic increment on `result_pos`. Subsequently, each thread with
 *  a selected value and `valid_write` writes the result into the reserved range.
 *  Only full warps are allowed to enter this function.
 *
 *  \param x the value for each thread
 *  \param result result range.
 *  \param result_pos pointer to an integer holding the ID of the first
 *      written result element by this function (usually initialized
 *      with zero). This counter must be shared among all threads, which
 *      enter this function because it is incremented and read atomically.
 *  \param lane_id ID of the thread within the warp.
 *  \param valid_write some threads might be disabled with this flag (useful for
 *      handling range borders).
 *  \param select_op lambda to select the values.
 */
template <typename T, typename It, typename Bool, class F>
CUDA_FD void select_if_warp_aggregated(T x,
                                       It selected_values,
                                       int* selected_values_pos,
                                       int lane_id,
                                       Bool valid_write,
                                       F select_op) {

	const bool p = select_op(x) && valid_write;

	const unsigned umask = __ballot_sync(0xffffffff, p);
	const int num_selected_values = __popc(umask);

	constexpr int source_lane = 0;
	int pos;
	if (lane_id == source_lane) {
		pos = atomicAdd(selected_values_pos, num_selected_values);
	}
	const unsigned umask_before_me =
	    (~unsigned(0)) >> (sizeof(unsigned) * 8 - lane_id);
	const int num_selected_values_before_me = __popc(umask & umask_before_me);
	pos = __shfl_sync(0xffffffff, pos, source_lane);
	pos += num_selected_values_before_me;
	if (p) {
		selected_values[pos] = x;
	}
}

/*! \brief select_if as a block level primitive.
 *
 *  \param values iterator to range of length N
 *  \param N length of input values
 *  \param selected_values iterator to range of length `*selected_values_pos + N`
 *  \param selected_values_pos pointer to an integer holding the ID of the first
 *      written selected_values element by this function (usually initialized
 *      with zero). This counter must be shared among all threads, which
 *      enter this function because it is incremented and read atomically.
 *  \param tid thread ID
 *  \param num_threads number of threads which enter this function.
 *  \param select_op lambda to select the values.
 */
template <typename It, typename ItResult, typename I0, typename I1, class F>
CUDA_FD void select_if(It values,
                       I0 N,
                       ItResult selected_values,
                       int* selected_values_pos,
                       int tid,
                       I1 num_threads,
                       F select_op) {

	const int num_tiles = thrustshift::ceil_divide(N, num_threads);
	for (int tile_id = 0; tile_id < num_tiles - 1; ++tile_id) {
		const int j = tile_id * num_threads + tid;
		auto x = values[j];
		select_if_warp_aggregated(x,
		                          selected_values,
		                          selected_values_pos,
		                          tid % warp_size,
		                          std::bool_constant<true>(),
		                          select_op);
	}
	auto rest = N % num_threads;
	if (rest > 0) {
		const int j = (num_tiles - 1) * num_threads + tid;
		const bool valid_rw = tid < rest;
		using T = typename std::remove_const<
		    typename std::iterator_traits<It>::value_type>::type;
		const auto x = [&]() -> T {
			if (valid_rw) {
				return values[j];
			}
			return T{};
		}();
		select_if_warp_aggregated(x,
		                          selected_values,
		                          selected_values_pos,
		                          tid % warp_size,
		                          valid_rw,
		                          select_op);
	}
}

} // namespace implicit_unroll

} // namespace device_function

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
