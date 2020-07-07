#pragma once

#include <algorithm>
#include <cmath>

namespace thrustshift {

template <class Range>
auto count_finite_values(Range&& r) {
	return std::count_if(
	    r.begin(), r.end(), [](const auto& e) { return std::isfinite(e); });
}

template <class Range>
double fraction_of_finite_values(Range&& r) {
	return (double) count_finite_values(r) / r.size();
}

//! |a - b| / |b|
template<typename T>
double relative_difference(T a, T b) {
	return std::abs(a - b) / std::abs(b);
}

//! Divide first and ceil second (a / b)
template <typename T>
T ceil_divide(T a, T b) {
	return a / b + (a % b == 0 ? 0 : 1);
}


} // namespace thrustshift
