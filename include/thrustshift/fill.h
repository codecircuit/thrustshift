#pragma once

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/define_specifiers.hpp>
#include <cuda/runtime_api.hpp>

#include <thrustshift/math.h>

namespace thrustshift {

namespace device_function {

namespace explicit_unroll {

/*! \brief Fill range of length `N` with `x`.
 *
 *  \param N length of the range
 *  \param p pointer to the range
 *  \param x value
 *  \param tid ID of the thread which enters the function
 *  \param num_threads total amount of threads which enter the function
 */
template <typename T, int num_threads, int N>
CUDA_FD void fill(T* p, T x, int tid) {
	constexpr int num_elements_per_thread = N / num_threads;
#pragma unroll
	for (int i = 0; i < num_elements_per_thread; ++i) {
		p[i * num_threads + tid] = x;
	}
	constexpr int num_rest = N % num_threads;
	if (tid < num_rest) {
		p[num_elements_per_thread * num_threads + tid] = x;
	}
}

} // namespace explicit_unroll

namespace implicit_unroll {

template <typename T0, typename T1, typename I0, typename I1, typename I2>
CUDA_FD void fill(T0* p, T1 x, I0 tid, I1 num_threads, I2 N) {
	auto num_elements_per_thread = N / num_threads;
#pragma unroll
	for (int i = 0; i < num_elements_per_thread; ++i) {
		p[i * num_threads + tid] = x;
	}
	auto num_rest = N % num_threads;
	if (tid < num_rest) {
		p[num_elements_per_thread * num_threads + tid] = x;
	}
}

} // namespace implicit_unroll

} // namespace device_function

namespace kernel {
template <typename T, class Range>
__global__ void fill(Range r, T val) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < r.size()) {
		r[gtid] = val;
	}
}

} // namespace kernel

namespace async {

template <typename T, class Range>
void fill(cuda::stream_t& stream, Range&& r, T val) {

	constexpr cuda::grid::block_dimension_t block_dim = 256;
	const cuda::grid::dimension_t grid_dim =
	    ceil_divide(r.size(), gsl_lite::narrow<decltype(r.size())>(block_dim));

	using RangeT = typename std::remove_reference<Range>::type::value_type;

	if (!r.empty()) {
		cuda::enqueue_launch(kernel::fill<T, gsl_lite::span<RangeT>>,
		                     stream,
		                     cuda::make_launch_config(grid_dim, block_dim),
		                     r,
		                     val);
	}
}

} // namespace async

} // namespace thrustshift
