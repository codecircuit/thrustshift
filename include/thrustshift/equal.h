#pragma once

#include <thrust/execution_policy.h>
#include <thrust/equal.h>

namespace thrustshift {

template<class RangeA, class RangeB>
bool equal(RangeA&& a, RangeB&& b) {
	gsl_Expects(a.size() == b.size());
	return thrust::equal(thrust::cuda::par, a.begin(), a.end(), b.begin());
}

}
