#pragma once

#include <type_traits>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

template <class T, class RangeOfRanges>
std::vector<gsl_lite::span<T>> make_spans_from_ranges(RangeOfRanges&& r) {
	std::vector<gsl_lite::span<T>> spans(r.size());
	for (size_t i = 0; i < spans.size(); ++i) {
		spans[i] = gsl_lite::make_span(r[i].data(), r[i].size());
	}
	return spans;
}

template <class T, class RangeOfPtrs>
std::vector<gsl_lite::span<T>> make_spans_from_ptrs(RangeOfPtrs&& r, size_t N) {

	std::vector<gsl_lite::span<T>> spans(r.size());
	for (size_t i = 0; i < spans.size(); ++i) {
		spans[i] = gsl_lite::make_span(r[i].get(), N);
	}
	return spans;
}

} // namespace thrustshift
