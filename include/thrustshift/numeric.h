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

} // namespace thrustshift
