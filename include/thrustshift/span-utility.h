#pragma once

#include <type_traits>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

/*! \brief Make multiple spans from a range of objects which contain ranges.
 *
 *  \param r
 *  \param get_range This functor must **not** return a copy of any object
 *
 *  ```cpp
 *  struct S {
 *
 *    std::vector<int> v;
 *  };
 *  ...
 *
 *  std::vector<S> ss;
 *  // fill ss
 *
 *  //// WRONG
 *  auto spans_with_invalid_ptrs = make_spans_from_ranges(ss, [](const auto& e) { return e.v; });
 *
 *  //// CORRECT
 *  auto spans = make_spans_from_ranges(ss, [](const auto& e) { return gsl_lite::make_span(e.v); });
 *  ```
 *
 *  The wrong lambda is returning a copy of member vector `v`, which is then deallocated.
 *
 */
template <class T, class RangeOfRanges, class GetRangeCallback>
std::vector<gsl_lite::span<T>> make_spans_from_ranges(
    RangeOfRanges&& r,
    GetRangeCallback&& get_range) {
	std::vector<gsl_lite::span<T>> spans(r.size());
	for (size_t i = 0; i < spans.size(); ++i) {
		spans[i] =
		    gsl_lite::make_span(get_range(r[i]).data(), get_range(r[i]).size());
	}
	return spans;
}

template <class T, class RangeOfRanges>
std::vector<gsl_lite::span<T>> make_spans_from_ranges(RangeOfRanges&& r) {
	auto get_range = [](auto& c) { return gsl_lite::make_span(c); };
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
