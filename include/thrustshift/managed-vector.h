#pragma once

#include <vector>

#include <thrustshift/managed-allocator.h>

namespace thrustshift {

template <class T, class Allocator = managed_allocator<T>>
using managed_vector = std::vector<T, Allocator>;

//! If a span-like class does not provide `begin()` and `end()` functions, but
//! offers a `operator[]` and a `size()` function, like the vectors classes of
//! Eigen3
template<class Range>
auto make_managed_vector(Range&& r) {

	using value_type = typename Range::value_type;
	managed_vector<value_type> result;
	result.reserve(r.size());
	for (size_t i = 0; i < r.size(); ++i) {
		result.push_back(r[i]);
	}
	return result;
}

}
