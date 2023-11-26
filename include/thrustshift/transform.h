#pragma once

#include <type_traits>
#include <utility>

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/defines.h>

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
void transform(cudaStream_t& stream,
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

	constexpr unsigned block_dim = 128;
	const unsigned grid_dim = (src.size() + block_dim - 1) / block_dim;
	kernel::transform<src_value_type, dst_value_type, decltype(f)>
	    <<<grid_dim, block_dim, 0, stream>>>(src, dst, f);
	THRUSTSHIFT_CHECK_CUDA_ERROR(cudaGetLastError());
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
THRUSTSHIFT_FHD auto transform_impl(const Arr<T, N>& arr,
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
THRUSTSHIFT_FHD auto transform(const Arr<T, N>& arr, F&& f) {
	return detail::transform_impl(arr, std::forward<F>(f), Indices{});
}

} // namespace array

namespace tuple {

namespace detail {

template <typename Tuple, typename F, std::size_t... I>
THRUSTSHIFT_FHD auto transform_impl(Tuple&& t,
                                    F&& f,
                                    std::index_sequence<I...>) {
	using TupleT = typename std::remove_reference<Tuple>::type;
	using std::get;
	return TupleT(f(get<I>(t))...);
}

} // namespace detail

template <typename... Ts, template <typename...> class TupleT, typename F>
THRUSTSHIFT_FHD auto transform(TupleT<Ts...> const& t, F&& f) {
	using std::tuple_size;
	auto seq = std::make_index_sequence<tuple_size<TupleT<Ts...>>::value>{};
	return detail::transform_impl(t, std::forward<F>(f), seq);
}

} // namespace tuple

} // namespace thrustshift
