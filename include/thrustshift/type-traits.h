#pragma once

#include <type_traits>
#include <utility>

#include <thrust/tuple.h>

namespace thrustshift {

template <typename T, std::size_t N, typename... REST>
struct make_thrust_tuple_type {
	using type = typename make_thrust_tuple_type<T, N - 1, T, REST...>::type;
};

template <typename T, typename... REST>
struct make_thrust_tuple_type<T, 0, REST...> {
	using type = thrust::tuple<REST...>;
};

template <typename T, T... Ints>
constexpr std::size_t get_integer_sequence_length(
    std::integer_sequence<T, Ints...>) noexcept {
	return (... + (std::size_t(1) + std::size_t(0 * Ints)));
}

namespace detail {

template <class T>
struct is_tuple {
	constexpr static bool value = false;
};

template <class... Ts>
struct is_tuple<thrust::tuple<Ts...>> {
	constexpr static bool value = true;
};

template <class... Ts>
struct is_tuple<std::tuple<Ts...>> {
	constexpr static bool value = true;
};

} // namespace detail

template <typename T>

//using is_tuple = detail::is_tuple<typename std::decay<T>::type>;
using is_tuple = detail::is_tuple<typename std::decay<T>::type>;


template<typename T>
struct make_uintegral_of_equal_size {
	static_assert(std::is_integral<T>::value);
	using type =  typename std::make_unsigned<T>::type;
};

template<>
struct make_uintegral_of_equal_size<float> {
	using type = uint32_t;
};

template<>
struct make_uintegral_of_equal_size<double> {
	using type = uint64_t;
};

} // namespace thrustshift
