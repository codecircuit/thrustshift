#pragma once

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/defines.h>
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
THRUSTSHIFT_FD void fill(T* p, T x, int tid) {
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
THRUSTSHIFT_FD void fill(T0* p, T1 x, I0 tid, I1 num_threads, I2 N) {
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
void fill(cudaStream_t& stream, Range&& r, T val) {

	constexpr unsigned block_dim = 256;
	const unsigned grid_dim =
	    ceil_divide(r.size(), gsl_lite::narrow<decltype(r.size())>(block_dim));

	using RangeT = typename std::remove_reference<Range>::type::value_type;

	if (!r.empty()) {
		kernel::fill<T, gsl_lite::span<RangeT>>
		    <<<grid_dim, block_dim, 0, stream>>>(r, val);
		THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
	}
}

} // namespace async

} // namespace thrustshift
