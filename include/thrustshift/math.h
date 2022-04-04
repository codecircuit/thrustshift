#pragma once

#include <cmath>
#include <exception>
#include <type_traits>

#include <cuda/define_specifiers.hpp>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace detail {

template <typename I, std::enable_if_t<std::is_signed<I>::value, bool> = true>
CUDA_FHD I abs_(I i) {
	return i < I(0) ? I(-1) * i : i;
}

// To prevent compiler warnings about unnecassary comparisons of unsigned types
// and zero
template <typename I, std::enable_if_t<std::is_unsigned<I>::value, bool> = true>
CUDA_FHD I abs_(I i) {
	return i;
}

} // namespace detail

/*! \brief divide and ceil two integral numbers.
 *
 *  Examples:
 *
 *  ```cpp
 *  ceil_divide(10, 10) = 1
 *  ceil_divide(10, 11) = 1
 *  ceil_divide(-10, 11) = -1
 *  ceil_divide(10, 9) = 2
 *  ceil_divide(10, 5) = 2
 *  ceil_divide(10, 4) = 3
 *  ceil_divide(0, 0) = 0
 *  ceil_divide(0, 10) = 0
 *  ceil_divide(10, 0) // error
 *  ceil_divide(-10, 6) = -2
 *  ```
 */
template <typename I, std::enable_if_t<std::is_integral<I>::value, bool> = true>
CUDA_FHD I ceil_divide(I a, I b) {
	auto sign_ = [](I x) { return x > I(0) ? I(1) : I(-1); };
	if (a == I(0)) {
		return 0;
	}
	gsl_Expects(b != I(0));

	// Do not use `std::abs` because there are no specializations for
	// unsigned types. Then the compiler does not know which overload it
	// should use if e.g. `ceil_divide(10llu, 5llu)` is called.
	if (detail::abs_(b) > detail::abs_(a)) {
		return I(1) * sign_(b) * sign_(a);
	}
	return a / b + (a % b == 0 ? 0 : (sign_(b) * sign_(a)));
}

//! Useful to sort data with resepect to their absolute values without changing the values
template <typename T>
struct AbsView {
	using value_type = T;
	T value;
};

template <typename T>
CUDA_FHD bool operator==(const AbsView<T>& a, const AbsView<T>& b) {
	using std::abs;
	return abs(a.value) == abs(b.value);
}

template <typename T>
CUDA_FHD bool operator!=(const AbsView<T>& a, const AbsView<T>& b) {
	return !(a == b);
}

template <typename T>
CUDA_FHD bool operator<(const AbsView<T>& a, const AbsView<T>& b) {
	using std::abs;
	return abs(a.value) < abs(b.value);
}

template <typename T>
CUDA_FHD bool operator>(const AbsView<T>& a, const AbsView<T>& b) {
	using std::abs;
	return abs(a.value) > abs(b.value);
}

template <typename T>
CUDA_FHD bool operator>=(const AbsView<T>& a, const AbsView<T>& b) {
	using std::abs;
	return abs(a.value) >= abs(b.value);
}

template <typename T>
CUDA_FHD bool operator<=(const AbsView<T>& a, const AbsView<T>& b) {
	using std::abs;
	return abs(a.value) <= abs(b.value);
}

/* \brief Calculate sinus and cosinus with fast CUDA math API.
 *
 * Usage example:
 * ```
 * auto [s, c] = thrustshift::sincos(x);
 * ```
 */
template <typename T>
CUDA_FHD std::tuple<T, T> sincos(T x) {
	T sin_result, cos_result;
#ifdef __CUDA_ARCH__
	if constexpr (std::is_same<T, double>::value) {
		::sincos(x, &sin_result, &cos_result);
	}
	else if constexpr (std::is_same<T, float>::value) {
		sincosf(x, &sin_result, &cos_result);
	}
#else
	sin_result = std::sin(x);
	cos_result = std::cos(x);
#endif
	return {sin_result, cos_result};
}

} // namespace thrustshift
