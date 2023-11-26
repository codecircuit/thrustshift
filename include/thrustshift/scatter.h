#pragma once

#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace kernel {

template <typename SrcT, typename MapT, typename DstT>
__global__ void scatter(gsl_lite::span<const SrcT> src,
                        gsl_lite::span<const MapT> map,
                        gsl_lite::span<DstT> dst) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < map.size()) {
		dst[map[gtid]] = src[gtid];
	}
}

} // namespace kernel

namespace async {

template <class SrcRange, class MapRange, class DstRange>
void scatter(cudaStream_t& stream,
             SrcRange&& src,
             MapRange&& map,
             DstRange&& dst) {

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

	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = (src.size() + block_dim - 1) / block_dim;
	kernel::scatter<src_value_type, map_index_type, dst_value_type>
	    <<<grid_dim, block_dim, 0, stream>>>(src, map, dst);
}

} // namespace async

} // namespace thrustshift
