#pragma once

#include <bitset>
#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

#include <thrustshift/defines.h>

namespace thrustshift {

namespace detail {

// to be compiler independent std::bitset is used here
template <typename I>
int count_leading_zeros_cpu(I i) {

	std::bitset<sizeof(I) * 8> bs(i);
	static_assert(bs.size() > 0);
	int lz = 0;
	for (int bi = gsl_lite::narrow<int>(sizeof(I) * 8) - 1; bi >= 0; --bi) {
		if (bs[bi]) {
			break;
		}
		++lz;
	}
	return lz;
}

template <typename I>
int count_leading_zeros_gpu(I i);

template <>
THRUSTSHIFT_FD int count_leading_zeros_gpu<int>(int i) {
	return __clz(i);
}

template <>
THRUSTSHIFT_FD int count_leading_zeros_gpu<long long int>(long long int i) {
	return __clzll(i);
}

} // namespace detail

template <typename I>
THRUSTSHIFT_FHD int count_leading_zeros(I i) {

#ifdef __CUDA_ARCH__
	return detail::count_leading_zeros_gpu(i);
#else
	return detail::count_leading_zeros_cpu(i);
#endif
}

} // namespace thrustshift
