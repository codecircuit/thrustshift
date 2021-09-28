#pragma once

#include <tuple>

#include <thrust/tuple.h>

namespace thrustshift {

template <class Tuple>
struct tuple_size;

template <typename... Ts>
struct tuple_size<std::tuple<Ts...>> {
	constexpr static std::size_t value =
	    std::tuple_size<std::tuple<Ts...>>::value;
};

template <typename... Ts>
struct tuple_size<thrust::tuple<Ts...>> {
	constexpr static std::size_t value =
	    thrust::tuple_size<thrust::tuple<Ts...>>::value;
};

} // namespace thrustshift
