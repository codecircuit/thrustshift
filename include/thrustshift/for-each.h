#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include <cuda/define_specifiers.hpp>

#include <thrustshift/tuple-traits.h>
#include <thrustshift/type-traits.h>

namespace thrustshift {

namespace tuple {

namespace detail {

template <typename Tuple, typename F, std::size_t... I>
CUDA_FHD void for_each_impl(Tuple&& t, F&& f, std::index_sequence<I...>) {
	using std::forward;
	using std::get;
	[[maybe_unused]] auto l = {(f(get<I>(forward<Tuple>(t))), 0)...};
}

} // namespace detail

template <class TupleT, typename F>
CUDA_FHD void for_each(TupleT&& t, F&& f) {
	auto seq = std::make_index_sequence<thrustshift::tuple_size<
	    typename std::remove_reference<TupleT>::type>::value>{};
	using std::forward;
	detail::for_each_impl(forward<TupleT>(t), forward<F>(f), seq);
}

template <std::size_t N, class TupleT, typename F>
CUDA_FHD void for_each_n(TupleT&& t, F&& f) {
	auto seq = std::make_index_sequence<N>{};
	using std::forward;
	detail::for_each_impl(forward<TupleT>(t), forward<F>(f), seq);
}

namespace detail {

template <typename TupleA, typename TupleB, typename F, std::size_t... I>
CUDA_FHD void for_each_impl(TupleA&& a,
                            TupleB&& b,
                            F&& f,
                            std::index_sequence<I...>) {
	using std::get;
	using std::forward;
	[[maybe_unused]] auto l = {(f(get<I>(forward<TupleA>(a)), get<I>(forward<TupleB>(b))), 0)...};
}

} // namespace detail

template <class TupleA, class TupleB, typename F>
CUDA_FHD void for_each(TupleA&& a, TupleB&& b, F&& f) {

	using TupleA_ = typename std::decay<TupleA>::type;
	using TupleB_ = typename std::decay<TupleB>::type;

	auto seq_a =
	    std::make_index_sequence<thrustshift::tuple_size<TupleA_>::value>{};
	auto seq_b =
	    std::make_index_sequence<thrustshift::tuple_size<TupleB_>::value>{};

	static_assert(get_integer_sequence_length(seq_a) ==
	              get_integer_sequence_length(seq_b));
	detail::for_each_impl(std::forward<TupleA>(a),
	                      std::forward<TupleB>(b),
	                      std::forward<F>(f),
	                      seq_a);
}

template <std::size_t N, class TupleA, class TupleB, typename F>
CUDA_FHD void for_each_n(TupleA&& a, TupleB&& b, F&& f) {

	auto seq = std::make_index_sequence<N>{};

	detail::for_each_impl(std::forward<TupleA>(a),
	                      std::forward<TupleB>(b),
	                      std::forward<F>(f),
	                      seq);
}

} // namespace tuple

} // namespace thrustshift
