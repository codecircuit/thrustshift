#pragma once

#include <type_traits>
#include <utility>

#include <cuda/define_specifiers.hpp>
#include <cuda/runtime_api.hpp>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace kernel {

template <typename SrcT, typename DstT, class F>
__global__ void transform(gsl_lite::span<const SrcT> src,
                          gsl_lite::span<DstT> dst,
                          F f) {

	const auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < src.size()) {
		dst[gtid] = f(src[gtid]);
	}
}

} // namespace kernel

namespace async {

template <class SrcRange, class DstRange, class UnaryFunctor>
void transform(cuda::stream_t& stream,
               SrcRange&& src,
               DstRange&& dst,
               UnaryFunctor f) {
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
	auto k = kernel::transform<src_value_type, dst_value_type, decltype(f)>;
	cuda::enqueue_launch(k, stream, c, src, dst, f);
}

} // namespace async

namespace array {

namespace detail {

template <typename T,
          std::size_t N,
          template <typename, std::size_t>
          class Arr,
          class F,
          std::size_t... I>
CUDA_FHD auto transform_impl(const Arr<T, N>& arr,
                             F&& f,
                             std::index_sequence<I...>) {
	return Arr<T, N>({f(arr[I])...});
}

} // namespace detail

template <typename T,
          std::size_t N,
          template <typename, std::size_t>
          class Arr,
          class F,
          typename Indices = std::make_index_sequence<N>>
CUDA_FHD auto transform(const Arr<T, N>& arr, F&& f) {
	return detail::transform_impl(arr, std::forward<F>(f), Indices{});
}

} // namespace array

} // namespace thrustshift
