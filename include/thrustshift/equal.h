#pragma once

#include <thrust/execution_policy.h>
#include <thrust/equal.h>

namespace thrustshift {

template<class RangeA, class RangeB>
bool equal(RangeA&& a, RangeB&& b) {
	// std::equal is undefined if range b is shorter than range a
	gsl_Expects(a.size() == b.size());
	return thrust::equal(thrust::cuda::par, a.begin(), a.end(), b.begin());
}

}
