#pragma once

#include <cuda/runtime_api.hpp>

namespace thrustshift {

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
	    (r.size() + block_dim - 1) / block_dim;

	using RangeT = typename std::remove_reference<Range>::type;

	cuda::enqueue_launch(kernel::fill<T, RangeT>,
	                     stream,
	                     cuda::make_launch_config(grid_dim, block_dim),
	                     r,
	                     val);
}

} // namespace async

} // namespace thrustshift
