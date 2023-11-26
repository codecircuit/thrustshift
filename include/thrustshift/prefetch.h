#pragma once

#include <type_traits>

#include <thrustshift/defines.h>

namespace thrustshift {

namespace async {

// device = device id of target device, device = cudaCpuDeviceId will prefetch to host
template <class Range>
void prefetch(cudaStream_t& stream, int device_id, Range&& r) {
	using T_ = typename std::remove_reference<Range>::type::value_type;
	// Due to missing const_range in cuda-api-wrappers
	using T = typename std::remove_const<T_>::type;

	THRUSTSHIFT_CHECK_CUDA_ERROR(
	    cudaMemPrefetchAsync(reinterpret_cast<const void*>(r.data()),
	                         r.size() * sizeof(T),
	                         device_id,
	                         stream));
}

template <class Range>
void prefetch_to_host(cudaStream_t& stream, Range&& r) {
	using T_ = typename std::remove_reference<Range>::type::value_type;
	// Due to missing const_range in cuda-api-wrappers
	using T = typename std::remove_const<T_>::type;
	prefetch(stream, cudaCpuDeviceId, r);
}

} // namespace async

} // namespace thrustshift
