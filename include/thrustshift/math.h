#pragma once
#include <cmath>
#include <exception>
#include <type_traits>

namespace thrustshift {

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
I ceil_divide(I a, I b) {
	auto sign_ = [](I x) { return x > I(0) ? I(1) : I(-1); };
	if (a == I(0)) {
		return 0;
	}
	if (b == I(0)) {
		throw std::domain_error("Divide by zero exception");
	}
	using std::abs;
	if (abs(b) > abs(a)) {
		return I(1) * sign_(b) * sign_(a);
	}
	return a / b + (a % b == 0 ? 0 : (sign_(b) * sign_(a)));
}

} // namespace thrustshift
