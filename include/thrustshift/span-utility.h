#pragma once

#include <type_traits>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

template <class T, class RangeOfRanges>
std::vector<gsl_lite::span<T>> make_spans_from_ranges(RangeOfRanges&& r) {
//	using T = typename std::remove_reference<RangeOfRanges>::type::value_type::value_type;
	std::vector<gsl_lite::span<T>> spans;
	spans.reserve(r.size());
	for (auto& e : r) {
		spans.emplace_back(gsl_lite::make_span(e.data(), e.size()));
	}
	return spans;
}

template <class T, class RangeOfPtrs>
std::vector<gsl_lite::span<T>> make_spans_from_ptrs(RangeOfPtrs&& r, size_t N) {

	std::vector<gsl_lite::span<T>> spans;
	spans.reserve(r.size());
	for (auto& e : r) {
		spans.emplace_back(gsl_lite::make_span(e.get(), N));
	}
	return spans;
}

} // namespace thrustshift
