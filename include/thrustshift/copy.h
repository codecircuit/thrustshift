#pragma once

#include <type_traits>

#include <cuda/runtime_api.hpp>
#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/fill.h>

namespace thrustshift {

namespace kernel {

template <typename SrcT, typename DstT>
__global__ void copy(gsl_lite::span<const SrcT> src, gsl_lite::span<DstT> dst) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		dst[gtid] = src[gtid];
	}
}

template <typename SrcT, typename DstT, typename T, typename I>
__global__ void copy_find(gsl_lite::span<const SrcT> src,
                          gsl_lite::span<DstT> dst,
                          T value,
                          I* pos) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		const auto src_value = src[gtid];
		dst[gtid] = src_value;
		if (src_value == value) {
			*pos = gtid;
		}
	}
}

} // namespace kernel

namespace async {

//! thrust uses sometimes a cudaMemcpyAsync instead of a copy kernel
template <class SrcRange, class DstRange>
void copy(cuda::stream_t& stream, SrcRange&& src, DstRange&& dst) {
	gsl_Expects(src.size() == dst.size());

	if (src.empty()) {
		return;
	}

	using src_value_type =
	    typename std::remove_reference<SrcRange>::type::value_type;
	using dst_value_type =
	    typename std::remove_reference<DstRange>::type::value_type;

	constexpr cuda::grid::block_dimension_t block_dim = 128;
	const cuda::grid::dimension_t grid_dim =
	    (src.size() + block_dim - 1) / block_dim;
	auto c = cuda::make_launch_config(grid_dim, block_dim);
	auto k = kernel::copy<src_value_type, dst_value_type>;
	cuda::enqueue_launch(k, stream, c, src, dst);
}

//! Copy and search for an element. If the element occurs more than once, it
//! is undefined which of the valid positions is returned. If the element does
//! not occur `pos` is unchanged.
template <class SrcRange, class DstRange, typename T, typename I>
void copy_find(cuda::stream_t& stream,
               SrcRange&& src,
               DstRange&& dst,
               const T& value,
               I* pos) {
	gsl_Expects(src.size() == dst.size());
	gsl_Expects(pos != nullptr);

	if (src.empty()) {
		return;
	}

	using src_value_type =
	    typename std::remove_reference<SrcRange>::type::value_type;
	using dst_value_type =
	    typename std::remove_reference<DstRange>::type::value_type;

	constexpr cuda::grid::block_dimension_t block_dim = 128;
	const cuda::grid::dimension_t grid_dim =
	    (src.size() + block_dim - 1) / block_dim;
	auto c = cuda::make_launch_config(grid_dim, block_dim);
	auto k = kernel::copy_find<src_value_type, dst_value_type, T, I>;
	cuda::enqueue_launch(k, stream, c, src, dst, value, pos);
}

} // namespace async

} // namespace thrustshift
