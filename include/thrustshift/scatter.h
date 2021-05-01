#pragma once

#include <type_traits>

#include <cuda/runtime_api.hpp>
#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace kernel {

template <typename SrcT, typename MapT, typename DstT>
__global__ void scatter(gsl_lite::span<const SrcT> src,
                        gsl_lite::span<const MapT> map,
                        gsl_lite::span<DstT> dst) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		dst[map[gtid]] = src[gtid];
	}
}

} // namespace kernel

namespace async {

template <class SrcRange, class MapRange, class DstRange>
void scatter(cuda::stream_t& stream,
             SrcRange&& src,
             MapRange&& map,
             DstRange&& dst) {

	gsl_Expects(src.size() == dst.size());
	gsl_Expects(src.size() == map.size());
	gsl_Expects(src.data() != dst.data());

	if (src.empty()) {
		return;
	}

	using src_value_type =
	    typename std::remove_reference<SrcRange>::type::value_type;
	using map_index_type =
	    typename std::remove_reference<MapRange>::type::value_type;
	using dst_value_type =
	    typename std::remove_reference<DstRange>::type::value_type;

	constexpr cuda::grid::block_dimension_t block_dim = 128;
	const cuda::grid::dimension_t grid_dim =
	    (src.size() + block_dim - 1) / block_dim;
	auto c = cuda::make_launch_config(grid_dim, block_dim);
	auto k = kernel::scatter<src_value_type, map_index_type, dst_value_type>;
	cuda::enqueue_launch(k, stream, c, src, map, dst);
}

} // namespace async

} // namespace thrustshift
