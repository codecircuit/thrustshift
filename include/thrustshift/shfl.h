#pragma once

#include <cuda/define_specifiers.hpp>
#include <gsl-lite/gsl-lite.hpp>

namespace thrustshift {

namespace warp_primitive {

//! With support for non-arithmetic types of size 2*sizeof(uint64_t)
template <typename T>
CUDA_FD T
shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize) {

	if constexpr (std::is_arithmetic<T>::value) {
		return __shfl_up_sync(mask, var, delta, width);
	}
	else {
		if constexpr (sizeof(T) == 2 * sizeof(uint64_t)) {
			uint64_t* ptr = reinterpret_cast<uint64_t*>((void*) &var);
			uint64_t a = ptr[0];
			uint64_t b = ptr[1];
			auto other_a = __shfl_up_sync(mask, a, delta, width);
			auto other_b = __shfl_up_sync(mask, b, delta, width);
			T result;
			uint64_t* ptr_result = reinterpret_cast<uint64_t*>((void*) &result);
			ptr_result[0] = other_a;
			ptr_result[1] = other_b;
			return result;
		}
	}
	static_assert(sizeof(T) == 2 * sizeof(uint64_t) ||
	              std::is_arithmetic<T>::value);
}

} // namespace warp_primitive

} // namespace thrustshift
