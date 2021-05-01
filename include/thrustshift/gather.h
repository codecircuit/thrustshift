#pragma once

#include <type_traits>

#include <cuda/runtime_api.hpp>
#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace kernel {

template <typename MapT, typename SrcT, typename DstT>
__global__ void gather(gsl_lite::span<const MapT> map, gsl_lite::span<const SrcT> src, gsl_lite::span<DstT> dst) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		dst[gtid] = src[map[gtid]];
	}
}

} // namespace kernel

namespace async {

template <class MapRange, class SrcRange, class DstRange>
void gather(cuda::stream_t& stream, MapRange&& map, SrcRange&& src, DstRange&& dst) {
	gsl_Expects(src.size() == dst.size());
	gsl_Expects(src.size() == map.size());
	gsl_Expects(src.data() != dst.data());

	if (src.empty()) {
		return;
	}

	using map_index_type =
	    typename std::remove_reference<MapRange>::type::value_type;
	using src_value_type =
	    typename std::remove_reference<SrcRange>::type::value_type;
	using dst_value_type =
	    typename std::remove_reference<DstRange>::type::value_type;

	constexpr cuda::grid::block_dimension_t block_dim = 128;
	const cuda::grid::dimension_t grid_dim =
	    (src.size() + block_dim - 1) / block_dim;
	auto c = cuda::make_launch_config(grid_dim, block_dim);
	auto k = kernel::gather<map_index_type, src_value_type, dst_value_type>;
	cuda::enqueue_launch(k, stream, c, map, src, dst);
}

} // namespace async

} // namespace thrustshift
