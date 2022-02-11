#pragma once

#include <gsl-lite/gsl-lite.hpp>

#include <cuda/runtime_api.hpp>

#include <thrustshift/math.h>

namespace thrustshift {

namespace kernel {

template <class Range, typename I>
__global__ void iota(Range r, I i0) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < r.size()) {
		r[gtid] = i0 + gtid;
	}
}

} // namespace kernel

// No conversion to gsl_lite::span in this namespace
namespace range {

namespace async {

template <class Range, typename I>
void iota(cuda::stream_t& stream, Range r, I i0) {

	constexpr int block_dim = 128;
	auto c = cuda::make_launch_config(
	    ceil_divide<decltype(r.size())>(r.size(), block_dim), block_dim);
	cuda::enqueue_launch(kernel::iota<Range, I>, stream, c, r, i0);
}

} // namespace async

} // namespace range

} // namespace thrustshift
