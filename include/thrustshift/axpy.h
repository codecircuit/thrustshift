#pragma once

#include <gsl/gsl-lite.hpp>

namespace thrustshift {

namespace kernel {

template <typename A, class RangeX, class RangeY, class RangeD>
__global__ void axpy(A a, RangeX x, RangeY y, RangeD d) {
	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	gsl_Expects(x.size() == y.size());
	gsl_Expects(y.size() == d.size());
	if (gtid < x.size()) {
		d[gtid] = a * x[gtid] + y[gtid];
	}
}

} // namespace kernel

namespace async {

template <typename A, class RangeX, class RangeY, class RangeD>
void axpy(cudaStream_t& stream, A a, RangeX&& x, RangeY&& y, RangeD&& d) {

	gsl_Expects(x.size() == y.size());
	gsl_Expects(y.size() == d.size());
	constexpr unsigned block_dim = 256;
	const unsigned grid_dim = (x.size() + block_dim - 1) / block_dim;

	using RangeXD = typename std::remove_reference<RangeX>::type;
	using RangeYD = typename std::remove_reference<RangeY>::type;
	using RangeDD = typename std::remove_reference<RangeD>::type;

	kernel::axpy<A, RangeXD, RangeYD, RangeDD>
	    <<<grid_dim, block_dim, 0, stream>>>(a, x, y, d);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace async

} // namespace thrustshift
