#pragma once

#include <cuda/runtime_api.hpp>
#include <gsl/gsl-lite.hpp>

namespace thrustshift {

namespace kernel {

template <typename A, class RangeX, class RangeY, class RangeD>
__global__ void axmy(A a, RangeX x, RangeY y, RangeD d) {
	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	gsl_Expects(x.size() == y.size());
	gsl_Expects(y.size() == d.size());
	if (gtid < x.size()) {
		d[gtid] = a * x[gtid] * y[gtid];
	}
}

} // namespace kernel

namespace async {

template <typename A, class RangeX, class RangeY, class RangeD>
void axmy(cuda::stream_t& stream, A a, RangeX&& x, RangeY&& y, RangeD&& d) {

	gsl_Expects(x.size() == y.size());
	gsl_Expects(y.size() == d.size());
	constexpr cuda::grid::block_dimension_t block_dim = 256;
	const cuda::grid::dimension_t grid_dim =
	    (x.size() + block_dim - 1) / block_dim;

	using RangeXD = typename std::remove_reference<RangeX>::type;
	using RangeYD = typename std::remove_reference<RangeY>::type;
	using RangeDD = typename std::remove_reference<RangeD>::type;

	cuda::enqueue_launch(kernel::axmy<A, RangeXD, RangeYD, RangeDD>,
	                     stream,
	                     cuda::make_launch_config(grid_dim, block_dim),
	                     a,
	                     x,
	                     y,
	                     d);
}

} // namespace async

} // namespace thrustshift
