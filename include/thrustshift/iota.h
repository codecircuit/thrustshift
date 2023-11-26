#pragma once

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/defines.h>
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
void iota(cudaStream_t& stream, Range r, I i0) {

	constexpr unsigned block_dim = 128;
	const unsigned grid_dim =
	    ceil_divide<decltype(r.size())>(r.size(), block_dim);

	kernel::iota<Range, I><<<grid_dim, block_dim, 0, stream>>>(r, i0);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace async

} // namespace range

} // namespace thrustshift
