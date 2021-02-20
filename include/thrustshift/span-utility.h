#pragma once

#include <type_traits>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

template <class T,
          class RangeOfRanges,
          class GetRangeCallback>
std::vector<gsl_lite::span<T>> make_spans_from_ranges(
    RangeOfRanges&& r,
    GetRangeCallback&& get_range) {
	std::vector<gsl_lite::span<T>> spans(r.size());
	for (size_t i = 0; i < spans.size(); ++i) {
		spans[i] = gsl_lite::make_span(get_range(r[i]).data(), get_range(r[i]).size());
	}
	return spans;
}

template <class T,
          class RangeOfRanges>
std::vector<gsl_lite::span<T>> make_spans_from_ranges(RangeOfRanges&& r) {
    auto get_range = [](auto& c) {
	        return gsl_lite::make_span(c);
	};
    return make_spans_from_ranges<T>(std::forward<RangeOfRanges>(r), get_range);
}

template <class T, class RangeOfPtrs>
std::vector<gsl_lite::span<T>> make_spans_from_ptrs(RangeOfPtrs&& r, size_t N) {

	std::vector<gsl_lite::span<T>> spans(r.size());
	for (size_t i = 0; i < spans.size(); ++i) {
		spans[i] = gsl_lite::make_span(r[i].get(), N);
	}
	return spans;
}

template <typename T>
gsl_lite::span<T> subtract(gsl_lite::span<T>& pool, size_t size_of_new_span) {
	gsl_Expects(pool.size() >= size_of_new_span);
	auto piece = pool.first(size_of_new_span);
	pool = pool.subspan(size_of_new_span);
	return piece;
}

} // namespace thrustshift
